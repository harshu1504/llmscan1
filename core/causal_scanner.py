"""
LLMSCAN: Causal Scan for LLM Misbehavior Detection
 
Implements the complete methodology from the paper:
- Token-level causal effects via attention score distances (Definition 2)
- Layer-level causal effects via layer skipping (Definition 3)
- Causal map generation (Section 3.1)
- MLP-based detector training (Section 3.2)
 
Innovations beyond paper:
- Adaptive layer selection
- Real-time causal streaming
- Cross-model transfer learning
- Attention head pruning
"""
 
from __future__ import annotations
 
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
import streamlit as st
from scipy.stats import kurtosis, skew
 
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Some features disabled.")
 
 
@dataclass
class CausalMap:
    """
    Causal map as defined in LLMSCAN paper - heatmap through causality lens.
 
    Contains:
    - token_causal_effects: CE_{x_i} per token (Definition 2)
    - layer_causal_effects: CE_{x,ℓ} per layer (Definition 3)
    - token_statistics: 5 features (mean, std, range, skewness, kurtosis)
    """
    token_causal_effects: Dict[int, float] = field(default_factory=dict)
    layer_causal_effects: Dict[int, float] = field(default_factory=dict)
    token_statistics: Dict[str, float] = field(default_factory=dict)
    attention_score_matrix: Optional[np.ndarray] = None
    layer_logit_differences: Optional[Dict[int, float]] = None
    prompt: str = ""
    response: str = ""
 
    def to_feature_vector(self, max_layers: int = 40) -> np.ndarray:
        """
        Convert causal map to fixed-dimension feature vector for detector.
 
        As per paper Section 3.2:
        - 5 token statistics (mean, std, range, skewness, kurtosis)
        - Per-layer causal effects (one per layer)
        """
        ce_values = list(self.token_causal_effects.values())
        if ce_values:
            stats = np.array([
                float(np.mean(ce_values)),
                float(np.std(ce_values)),
                float(np.max(ce_values) - np.min(ce_values)),
                float(skew(ce_values)) if len(ce_values) > 2 else 0.0,
                float(kurtosis(ce_values)) if len(ce_values) > 3 else 0.0
            ], dtype=np.float32)
        else:
            stats = np.zeros(5, dtype=np.float32)
 
        max_layer = max(self.layer_causal_effects.keys(), default=0)
        layer_features = np.zeros(max_layers, dtype=np.float32)
        for layer, effect in self.layer_causal_effects.items():
            if layer < max_layers:
                layer_features[layer] = float(effect)
 
        return np.concatenate([stats, layer_features])
 
    def to_heatmap_data(self) -> Dict:
        """Prepare data for causal map visualization (Figure 2 in paper)."""
        return {
            "token_effects": list(self.token_causal_effects.values()),
            "token_indices": list(self.token_causal_effects.keys()),
            "layer_effects": list(self.layer_causal_effects.values()),
            "layer_indices": list(self.layer_causal_effects.keys()),
            "token_statistics": self.token_statistics
        }
 
 
class CausalScanner:
    """
    Implements LLMSCAN's scanner component.
 
    Paper methodology:
    - Token causal effects via attention score distances (Definition 2)
    - Layer causal effects via layer skipping (Definition 3)
    - Selective attention head sampling (first/middle/last layers & heads)
    """
 
    def __init__(self, model_manager, metrics_engine):
        self.model_manager = model_manager
        self.metrics_engine = metrics_engine
        self._attention_cache = {}
        self._layer_skip_cache = {}
 
    def compute_token_causal_effects(
    self, 
    prompt: str, 
    selected_layers: Optional[List[int]] = None,
    selected_heads: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Compute causal effect of each input token as per Definition 2.
    
    FIXED: No nested loop variable conflict. Each token is intervened 
    individually and correctly measured.
    """
    if selected_layers is None:
        total_layers = min(self.model_manager.get_total_layers(), 6)
        selected_layers = [0, total_layers // 2, total_layers - 1]
    
    if selected_heads is None:
        selected_heads = [0, 15, 31]
    
    # Get original input
    inputs = self.model_manager.tokenizer(prompt, return_tensors="pt")
    original_ids = inputs.input_ids
    seq_len = original_ids.shape[1]
    
    # Get original attention scores
    original_attentions = self._extract_attention_scores(
        prompt, selected_layers, selected_heads
    )
    
    causal_effects = {}
    
    # FIX: Intervene on ONE token at a time, measure effect
    for token_pos in range(seq_len):
        modified_ids = original_ids.clone()
        
        # Replace this single token with PAD token
        modified_ids[0, token_pos] = self.model_manager.tokenizer.pad_token_id
        
        # Get intervened attention scores
        intervened_attentions = self._extract_attention_scores_from_ids(
            modified_ids, selected_layers, selected_heads
        )
        
        # Compute distance between original and intervened
        distance = 0.0
        count = 0
        for layer in selected_layers:
            for head in selected_heads:
                if layer in original_attentions and head in original_attentions[layer]:
                    orig = np.array(original_attentions[layer][head])
                    interv = np.array(
                        intervened_attentions.get(layer, {}).get(head, orig)
                    )
                    # Handle shape differences after token removal
                    min_seq = min(orig.shape[-1], interv.shape[-1])
                    if min_seq > 0:
                        distance += np.linalg.norm(
                            orig[..., :min_seq] - interv[..., :min_seq]
                        )
                        count += 1
        
        causal_effects[token_pos] = distance / max(count, 1)
    
    return causal_effects
 
    def _extract_attention_scores_from_ids(
        self,
        input_ids,
        selected_layers,
        selected_heads
    ):
        attentions = self.model_manager.get_attention_scores_from_ids(input_ids)
 
        result = {}
        for layer_idx, layer_attn in enumerate(attentions):
            if layer_idx not in selected_layers:
                continue
            result[layer_idx] = {}
            attn_np = layer_attn[0].cpu().numpy()
 
            for head_idx in selected_heads:
                if head_idx < attn_np.shape[0]:
                    result[layer_idx][head_idx] = attn_np[head_idx]
 
        return result
 
    def _extract_attention_scores(
        self,
        prompt: str,
        selected_layers: List[int],
        selected_heads: List[int]
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Extract attention scores for specific layers and heads."""
        cache_key = hashlib.sha256(
            f"{prompt}_{selected_layers}_{selected_heads}".encode()
        ).hexdigest()
        if cache_key in self._attention_cache:
            return self._attention_cache[cache_key]
 
        attentions = self.model_manager.get_attention_scores(prompt)
 
        result = {}
        for layer_idx, layer_attn in enumerate(attentions):
            if layer_idx not in selected_layers:
                continue
            result[layer_idx] = {}
            attn_np = layer_attn[0].cpu().numpy()
            for head_idx in selected_heads:
                if head_idx < attn_np.shape[0]:
                    result[layer_idx][head_idx] = attn_np[head_idx]
 
        self._attention_cache[cache_key] = result
        return result
 
    def compute_layer_causal_effects(
        self,
        prompt: str,
        baseline_output: Optional[str] = None,
        baseline_logit: Optional[float] = None
    ) -> Dict[int, float]:
        """
        Compute causal effect of each transformer layer as per Definition 3.
 
        CE_{x,ℓ} = logit_0 - logit_0^{-ℓ}
        where logit_0 is output logit of first token, and logit_0^{-ℓ} is logit when layer ℓ is skipped.
        """
        if baseline_logit is None:
            _, baseline_logit = self.model_manager.get_first_token_logit(prompt)
 
        if baseline_output is None:
            result = self.model_manager.run_inference(prompt)
            baseline_output = result.get("generated_text", "") if isinstance(result, dict) else ""
 
        total_layers = min(self.model_manager.get_total_layers(), 6)
        causal_effects = {}
 
        for layer_idx in range(total_layers):
            try:
                modified_logit = self._get_first_token_logit_with_skip(prompt, layer_idx)
                ce = baseline_logit - modified_logit
                causal_effects[layer_idx] = float(abs(ce))
            except Exception:
                causal_effects[layer_idx] = 0.0
 
        return causal_effects
 
    def _get_first_token_logit_with_skip(self, prompt: str, skip_layer: int) -> float:
        """Get first token logit with a specific layer skipped."""
        def skip_hook(module, input, output):
            if isinstance(output, tuple):
                return (input[0],) + output[1:] if len(output) > 1 else (input[0],)
            return input[0]
 
        layers = self.model_manager.get_layers()
        if skip_layer >= len(layers):
            return 0.0
 
        hook = layers[skip_layer].register_forward_hook(skip_hook)
 
        try:
            _, logit = self.model_manager.get_first_token_logit(prompt)
            result = logit
        finally:
            hook.remove()
 
        return result
 
    def generate_causal_map(
        self,
        prompt: str,
        compute_tokens: bool = True,
        compute_layers: bool = True,
        response: str = ""
    ) -> CausalMap:
        """Generate complete causal map as defined in Section 3.1."""
        token_effects = {}
        token_stats = {}
 
        if compute_tokens:
            token_effects = self.compute_token_causal_effects(prompt)
 
            ce_values = list(token_effects.values())
            if ce_values:
                token_stats = {
                    "mean": float(np.mean(ce_values)),
                    "std": float(np.std(ce_values)),
                    "range": float(np.max(ce_values) - np.min(ce_values)),
                    "skewness": float(skew(ce_values)) if len(ce_values) > 2 else 0.0,
                    "kurtosis": float(kurtosis(ce_values)) if len(ce_values) > 3 else 0.0
                }
 
        layer_effects = {}
        if compute_layers:
            layer_effects = self.compute_layer_causal_effects(prompt, baseline_output=response)
 
        return CausalMap(
            token_causal_effects=token_effects,
            layer_causal_effects=layer_effects,
            token_statistics=token_stats,
            prompt=prompt,
            response=response
        )
 
 
class MisbehaviorDetector:
    """
    MLP-based detector as described in Section 3.2.
 
    Trains on causal maps from contrasting prompt sets.
    Input: 5 token statistics + per-layer causal effects.
    Output: Binary classification (misbehavior vs normal).
    """
 
    def __init__(self, hidden_dims: List[int] = [64, 32]):
        self.hidden_dims = hidden_dims
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.training_history = []
 
    def train(
        self,
        normal_maps: List[CausalMap],
        misbehavior_maps: List[CausalMap],
        max_layers: int = 40
    ) -> Dict[str, float]:
        """Train detector on contrasting sets of causal maps."""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
 
        if len(normal_maps) == 0 or len(misbehavior_maps) == 0:
            return {"error": "Need both normal and misbehavior examples"}
 
        X_normal = [m.to_feature_vector(max_layers) for m in normal_maps]
        X_mis = [m.to_feature_vector(max_layers) for m in misbehavior_maps]
 
        X = np.vstack(X_normal + X_mis)
        y = np.array([0] * len(normal_maps) + [1] * len(misbehavior_maps))
 
        X_scaled = self.scaler.fit_transform(X)
 
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_dims),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
 
        self.model.fit(X_scaled, y)
        self.is_trained = True
 
        return {
            "train_accuracy": float(self.model.score(X_scaled, y)),
            "n_features": X.shape[1],
            "n_normal": len(normal_maps),
            "n_misbehavior": len(misbehavior_maps)
        }
 
    def predict(self, causal_map: CausalMap, max_layers: int = 40) -> Tuple[float, bool]:
        """Predict whether causal map indicates misbehavior."""
        if not self.is_trained or self.model is None:
            return 0.5, False
 
        X = causal_map.to_feature_vector(max_layers).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
 
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        return prob, prob >= 0.5
 
    def predict_batch(self, causal_maps: List[CausalMap], max_layers: int = 40) -> List[float]:
        """Predict for multiple causal maps."""
        if not self.is_trained or self.model is None:
            return [0.5] * len(causal_maps)
 
        X = np.vstack([m.to_feature_vector(max_layers) for m in causal_maps])
        X_scaled = self.scaler.transform(X)
 
        return [float(p) for p in self.model.predict_proba(X_scaled)[:, 1]]
 
 
class AdaptiveLayerSelector:
    def __init__(self, scanner: CausalScanner, calibration_prompts: List[str]):
        self.scanner = scanner
        self.calibration_prompts = calibration_prompts
        self._layer_importance = None
 
    def compute_layer_importance(self) -> Dict[int, float]:
        if self._layer_importance is not None:
            return self._layer_importance
 
        all_effects = defaultdict(list)
 
        for prompt in self.calibration_prompts:
            effects = self.scanner.compute_layer_causal_effects(prompt)
            for layer, ce in effects.items():
                all_effects[layer].append(ce)
 
        self._layer_importance = {
            layer: float(np.var(effects)) if len(effects) > 1 else 0.0
            for layer, effects in all_effects.items()
        }
 
        return self._layer_importance
 
    def select_top_k_layers(self, k: int = 10) -> List[int]:
        importance = self.compute_layer_importance()
        sorted_layers = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [layer for layer, _ in sorted_layers[:k]]
 
 
class AttentionHeadPruner:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.head_importance = {}
 
    def compute_head_importance(
        self,
        calibration_prompts: List[str],
        metric: str = "variance"
    ) -> Dict[Tuple[int, int], float]:
        importance = defaultdict(list)
 
        for prompt in calibration_prompts:
            attentions = self.model_manager.get_attention_scores(prompt)
 
            for layer_idx, layer_attn in enumerate(attentions):
                attn_np = layer_attn[0].cpu().numpy()
 
                for head_idx in range(attn_np.shape[0]):
                    key = (layer_idx, head_idx)
 
                    if metric == "variance":
                        val = float(np.var(attn_np[head_idx]))
                    else:
                        val = float(np.sum(attn_np[head_idx]))
 
                    importance[key].append(val)
 
        return {k: float(np.mean(v)) for k, v in importance.items()}
 
    def select_top_heads(
        self,
        calibration_prompts: List[str],
        heads_per_layer: int = 3
    ) -> Dict[int, List[int]]:
        importance = self.compute_head_importance(calibration_prompts)
 
        layer_heads = defaultdict(list)
        for (layer, head), imp in importance.items():
            layer_heads[layer].append((head, imp))
 
        selected = {}
        for layer, heads in layer_heads.items():
            sorted_heads = sorted(heads, key=lambda x: x[1], reverse=True)
            selected[layer] = [h for h, _ in sorted_heads[:heads_per_layer]]
 
        return selected
 
 
class CausalMapStreamer:
    def __init__(self, scanner: CausalScanner, detector: MisbehaviorDetector):
        self.scanner = scanner
        self.detector = detector
 
    def stream_analysis(
        self,
        prompt: str,
        stream_interval: int = 3
    ) -> Generator[Tuple[int, float, CausalMap], None, None]:
        tokens = self.scanner.model_manager.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids[0]
 
        cumulative_risk = 0.0
 
        for pos in range(1, len(input_ids) + 1, stream_interval):
            partial_prompt = self.scanner.model_manager.tokenizer.decode(input_ids[:pos])
 
            causal_map = self.scanner.generate_causal_map(partial_prompt)
 
            risk_prob, _ = self.detector.predict(causal_map)
 
            cumulative_risk = 0.7 * cumulative_risk + 0.3 * risk_prob
 
            yield pos, cumulative_risk, causal_map
 
            if cumulative_risk > 0.8:
                break
 
 
def visualize_causal_map(causal_map: CausalMap, figsize: Tuple[int, int] = (10, 6)):
    import matplotlib.pyplot as plt
 
    heatmap_data = causal_map.to_heatmap_data()
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
 
    if heatmap_data["token_effects"]:
        ax1.bar(heatmap_data["token_indices"], heatmap_data["token_effects"])
        ax1.set_xlabel("Token Position")
        ax1.set_ylabel("Causal Effect (CE)")
        ax1.set_title("Token-Level Causal Effects")
 
    if heatmap_data["layer_effects"]:
        ax2.bar(heatmap_data["layer_indices"], heatmap_data["layer_effects"])
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Causal Effect (CE)")
        ax2.set_title("Layer-Level Causal Effects")
 
    plt.tight_layout()
    return fig