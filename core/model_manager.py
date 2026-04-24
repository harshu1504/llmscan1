import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import hashlib
import json
import numpy as np
 
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
 
 
# ============================================================================
# INTENT-AWARE SAFETY CONFIGURATION
# ============================================================================
# Risk words are NOT enough on their own - they must appear with harmful intent
# signals (instructional framing, action verbs, target + method combos).
# Each entry: (risk_word, base_score, requires_instruction_framing)
RISK_LEXICON = {
    # These are HIGH-risk ONLY when paired with instructional framing
    "kill":      (4, True),
    "murder":    (4, True),
    "attack":    (4, True),
    "hack":      (4, True),
    "exploit":   (3, True),
    "steal":     (3, True),
    "bypass":    (3, True),
    "poison":    (3, True),
    "bomb":      (4, True),
    "explosive": (4, True),
    "weapon":    (3, True),
    "assault":   (3, True),
    "breach":    (2, True),
    # These score regardless of framing (always harmful in prompts)
    "child abuse":   (6, False),
    "csam":          (6, False),
    "bioweapon":     (6, False),
}
 
# Context words that CANCEL or REDUCE risk scores for ambiguous terms
SAFE_CONTEXT_SIGNALS = [
    # Medical / scientific
    "cancer", "disease", "illness", "patient", "treatment", "therapy",
    "doctor", "hospital", "medicine", "diagnosis", "symptom", "virus",
    "bacteria", "infection", "tumor", "surgery", "clinical",
    # Fictional / creative
    "story", "novel", "fiction", "character", "movie", "game",
    "plot", "scene", "narrative", "hypothetically", "imagine",
    # Academic / research
    "research", "study", "paper", "journal", "academic", "thesis",
    "history", "historical", "ancient", "war history", "documentary",
    # Legal / defensive
    "defend", "protect", "prevent", "security audit", "penetration test",
    "authorized", "legal", "law enforcement", "cybersecurity professional",
]
 
# Instructional intent phrases - these AMPLIFY risk when combined with risk words
INSTRUCTION_SIGNALS = [
    "how to", "how do i", "steps to", "guide to", "method to",
    "teach me", "show me how", "walk me through", "explain how",
    "give me instructions", "what are the steps", "ways to",
    "can you help me", "i want to", "i need to", "i plan to",
    "i'm trying to", "help me",
]
 
# Pure informational/curiosity signals - REDUCE risk amplification
INFO_SIGNALS = [
    "what is", "what are", "why is", "why does", "how does",
    "explain", "define", "describe", "tell me about", "is it",
    "are there", "can you explain", "what happens when",
]
 
 
class IntentAnalyzer:
    """
    Context-aware intent analysis that goes beyond keyword matching.
 
    Key insight: the same word ('dangerous', 'kill', 'hack') can appear in
    completely safe or harmful contexts. Intent is determined by:
    1. The framing (instructional vs informational vs narrative)
    2. The target (a person/system vs a concept/entity)
    3. Mitigating context (medical, fictional, academic, defensive)
    4. Specificity (generic curiosity vs operational planning)
    """
 
    def analyze(self, prompt: str) -> dict:
        p = prompt.lower().strip()
 
        # Step 1: Detect framing
        framing = self._detect_framing(p)
 
        # Step 2: Detect mitigating context
        safe_context_score = self._score_safe_context(p)
 
        # Step 3: Score risk words, with context awareness
        raw_risk, triggered_words, categories = self._score_risk_words(
            p, framing, safe_context_score
        )
 
        # Step 4: Apply context mitigation
        effective_risk = self._apply_mitigation(raw_risk, safe_context_score, framing)
 
        # Step 5: Determine intent label
        intent_label = self._label_intent(effective_risk, framing)
 
        return {
            "framing": framing,                    # "instructional" | "informational" | "narrative" | "neutral"
            "safe_context_score": safe_context_score,
            "raw_risk": raw_risk,
            "effective_risk": effective_risk,
            "triggered_words": triggered_words,
            "categories": categories,
            "intent_label": intent_label,           # "harmful" | "informational" | "neutral"
        }
 
    def _detect_framing(self, p: str) -> str:
        """Detect whether the prompt is seeking instructions, information, or is narrative."""
        instruction_hits = sum(1 for sig in INSTRUCTION_SIGNALS if sig in p)
        info_hits = sum(1 for sig in INFO_SIGNALS if sig in p)
 
        if instruction_hits >= 1:
            return "instructional"
        if info_hits >= 1:
            return "informational"
        if any(w in p for w in ["story", "novel", "fiction", "imagine", "character", "scene"]):
            return "narrative"
        return "neutral"
 
    def _score_safe_context(self, p: str) -> float:
        """
        Score how much safe/benign context is present.
        Returns 0.0 (no safe context) to 1.0 (strong safe context).
        """
        hits = sum(1 for ctx in SAFE_CONTEXT_SIGNALS if ctx in p)
        # Cap at 3 hits = full mitigation
        return min(hits / 3.0, 1.0)
 
    def _score_risk_words(
        self, p: str, framing: str, safe_context_score: float
    ) -> tuple:
        """
        Score risk words with framing awareness.
 
        Words that require instructional framing (requires_instruction=True)
        only contribute their full score when framing == 'instructional'.
        In informational framing they contribute half. In narrative, 25%.
        """
        raw_risk = 0
        triggered_words = []
        categories = set()
 
        framing_multipliers = {
            "instructional": 1.0,
            "neutral":       0.6,
            "informational": 0.3,  # "what is a hack?" is low risk
            "narrative":     0.15, # "the character hacks..." is very low risk
        }
        multiplier = framing_multipliers.get(framing, 0.6)
 
        for word, (base_score, requires_instruction) in RISK_LEXICON.items():
            if word in p:
                if requires_instruction:
                    score = base_score * multiplier
                else:
                    score = base_score  # Always harmful regardless of framing
                raw_risk += score
                triggered_words.append(word)
 
                # Assign category
                if word in ("kill", "murder", "attack", "assault"):
                    categories.add("violence")
                elif word in ("hack", "exploit", "bypass", "breach"):
                    categories.add("unauthorized_access")
                elif word in ("steal",):
                    categories.add("theft")
                elif word in ("bomb", "explosive", "weapon", "poison", "bioweapon"):
                    categories.add("weapons")
                elif word in ("child abuse", "csam"):
                    categories.add("child_safety")
 
        return raw_risk, triggered_words, list(categories)
 
    def _apply_mitigation(
        self, raw_risk: float, safe_context_score: float, framing: str
    ) -> float:
        """
        Reduce raw risk by safe context signals.
 
        Example: "is cancer dangerous?" triggers 'dangerous' (not in lexicon)
        but even if it did, 'cancer' as safe context would reduce the score.
 
        Example: "how to kill cancer cells?" - 'kill' triggers raw_risk,
        but 'cancer' + 'cells' provides strong safe context → effective_risk is low.
        """
        if safe_context_score >= 0.5:
            # Strong safe context: cut risk by up to 80%
            mitigation = 0.8 * safe_context_score
        elif safe_context_score > 0:
            # Partial safe context: cut risk proportionally
            mitigation = 0.5 * safe_context_score
        else:
            mitigation = 0.0
 
        # Informational framing with any safe context = strong mitigation
        if framing == "informational" and safe_context_score > 0:
            mitigation = max(mitigation, 0.75)
 
        effective_risk = raw_risk * (1.0 - mitigation)
        return max(effective_risk, 0.0)
 
    def _label_intent(self, effective_risk: float, framing: str) -> str:
        """Convert effective risk score to an intent label."""
        if effective_risk >= 4.0:
            return "harmful"
        if effective_risk >= 2.0 and framing == "instructional":
            return "suspicious"
        if framing == "informational" and effective_risk < 1.5:
            return "informational"
        if effective_risk < 1.0:
            return "neutral"
        return "suspicious"
 
 
class ModelManager:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.activations = {}
        self.hooks = []
        self._inference_cache = {}
        self._total_layers = None
        self._intent_analyzer = IntentAnalyzer()
 
    def load_model(self):
        if self.model is not None:
            return
        try:
            with st.spinner(f"Loading {self.model_name} on CPU..."):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
 
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                self._total_layers = len(self.get_layers())
                st.success(f"✅ Model loaded: {self.model_name} ({self._total_layers} layers)")
        except OSError as e:
            st.error(f"Model loading failed: {e}")
            try:
                fallback = "distilgpt2"
                with st.spinner(f"Loading fallback model {fallback}..."):
                    self.model_name = fallback
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self.model.eval()
                    self._total_layers = len(self.get_layers())
                st.info(f"Loaded fallback model: {fallback}")
            except Exception as e2:
                st.error(f"Fallback failed: {e2}")
                raise
 
    def get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'layers'):
            return self.model.base_model.layers
        else:
            return []
 
    def get_total_layers(self) -> int:
        if self._total_layers is None:
            self._total_layers = len(self.get_layers())
        return self._total_layers
 
    def attach_hooks(self):
        self.activations = {}
        self.remove_hooks()
 
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
 
        for i, layer in enumerate(self.get_layers()):
            self.hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))
 
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
 
    def attach_intervention_hooks(self, intervention):
        """
        Attach hooks with intervention support.
        intervention: dict {layer_idx: {'type': 'zero'|'scale'|'noise'|'skip', 'value': float}}
        'skip' implements paper Definition 3: bypass layer
        """
        self.activations = {}
        self.remove_hooks()
 
        def intervention_hook(name, config):
            def hook(model, input, output):
                if config['type'] == 'skip':
                    if isinstance(output, tuple):
                        return (input[0],) + output[1:] if len(output) > 1 else (input[0],)
                    return input[0]
 
                if isinstance(output, tuple):
                    val = output[0].clone()
                    rest = output[1:]
                else:
                    val = output.clone()
                    rest = None
 
                if config['type'] == 'zero':
                    val = torch.zeros_like(val)
                elif config['type'] == 'scale':
                    val = val * config.get('value', 0.5)
                elif config['type'] == 'noise':
                    val = val + torch.randn_like(val) * config.get('value', 0.1)
 
                self.activations[name] = val.detach()
 
                if rest:
                    return (val,) + rest
                return val
            return hook
 
        for i, layer in enumerate(self.get_layers()):
            if i in intervention:
                self.hooks.append(layer.register_forward_hook(
                    intervention_hook(f"layer_{i}", intervention[i])
                ))
            else:
                def capture_hook(name):
                    def hook(model, input, output):
                        if isinstance(output, tuple):
                            self.activations[name] = output[0].detach()
                        else:
                            self.activations[name] = output.detach()
                    return hook
                self.hooks.append(layer.register_forward_hook(capture_hook(f"layer_{i}")))
 
    def run_inference(self, prompt, intervention=None, max_new_tokens=50):
        """
        Run inference and return a result dict.
 
        Always returns:
          {
            "generated_text": str,
            "safety_results": dict,
            "trigger_info": dict | None,
            "hidden_states": tuple,
            "input_length": int,
            "generated_length": int,
          }
        """
        steered_prompt = f"{prompt}\nAnswer: " if "\nAnswer:" not in prompt else prompt
 
        try:
            ck = json.dumps(
                {'prompt': steered_prompt, 'intervention': intervention},
                sort_keys=True
            )
            cache_key = hashlib.sha256(ck.encode('utf-8')).hexdigest()
            if cache_key in self._inference_cache:
                return self._inference_cache[cache_key]
        except Exception:
            cache_key = None
 
        inputs = self.tokenizer(steered_prompt, return_tensors="pt").to(self.device)
 
        if intervention:
            self.attach_intervention_hooks(intervention)
        else:
            self.attach_hooks()
 
        generated_ids, trigger_info, hidden_states = self._generate_with_safety_monitoring(
            inputs.input_ids, steered_prompt, max_new_tokens
        )
 
        input_length = inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(
            generated_ids[0][input_length:], skip_special_tokens=True
        )
 
        safety_results = self._perform_safety_analysis(steered_prompt, hidden_states)
 
        result = {
            "generated_text": generated_text,
            "safety_results": safety_results,
            "trigger_info": trigger_info,
            "hidden_states": hidden_states,
            "input_length": input_length,
            "generated_length": len(generated_ids[0]) - input_length,
        }
 
        try:
            if cache_key:
                self._inference_cache[cache_key] = result
        except Exception:
            pass
 
        return result
 
    def get_attention_scores(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
 
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
 
        return outputs.attentions
 
    def get_first_token_logit(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
 
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
 
        return logits, float(torch.max(logits).item())
 
    def get_attention_scores_from_ids(self, input_ids):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True
            )
        return outputs.attentions
 
    # -------------------------------------------------------------------------
    # GENERATION WITH SAFETY MONITORING
    # -------------------------------------------------------------------------
 
    def _generate_with_safety_monitoring(self, input_ids, prompt, max_new_tokens=50):
        """
        Generate tokens with REAL-TIME safety monitoring.
        Stops generation IMMEDIATELY when risk is detected mid-generation.
        Returns: (generated_ids, trigger_info, hidden_states)
        """
        generated_ids = input_ids.clone()
        trigger_info = None
        hidden_states_at_stop = None
        
        # Pre-analyze prompt intent
        intent_result = self._intent_analyzer.analyze(prompt)
        
        # Track cumulative risk for the RESPONSE (not just prompt)
        response_risk = 0.0
        
        for step in range(max_new_tokens):
            outputs = self.model(
                input_ids=generated_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            token_str = self.tokenizer.decode(next_token[0])
            
            # ---- CRITICAL: Real-time risk detection on PARTIAL response ----
            response_so_far = self.tokenizer.decode(generated_ids[0])
            full_so_far = response_so_far + token_str
            
            # Check if the RESPONSE contains harmful content
            risk_score = self._compute_response_risk(full_so_far, intent_result)
            
            # ALSO check layer-level anomaly
            if outputs.hidden_states:
                layer_risk = self._compute_layer_anomaly_score(outputs.hidden_states)
                risk_score = max(risk_score, layer_risk * 0.5)
            
            if risk_score >= 0.7:
                trigger_info = {
                    "token": token_str,
                    "position": generated_ids.shape[1],
                    "risk_score": risk_score,
                    "step": step,
                    "layer_responsible": self._find_responsible_layer(outputs.hidden_states) if outputs.hidden_states else 0
                }
                hidden_states_at_stop = outputs.hidden_states
                break
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token[0].item() == self.tokenizer.eos_token_id:
                break
        
        if hidden_states_at_stop is None:
            hidden_states_at_stop = outputs.hidden_states if 'outputs' in dir() else ()
        
        return generated_ids, trigger_info, hidden_states_at_stop
    
    def _compute_response_risk(self, response: str, intent_result: dict) -> float:
        """Compute risk score for partial response (0-1 scale)."""
        r = response.lower()
        p_intent = intent_result.get("framing", "neutral")
        
        # Harmful response patterns (actual instructions being provided)
        harmful_response_markers = [
            "step 1", "step 2", "first,", "second,", "then,",
            "you need to", "you should", "here's how", "the way to",
            "password", "credentials", "bypass", "exploit", "hack into",
            "access without", "unauthorized", "vulnerability"
        ]
        
        # Calculate marker density
        marker_hits = sum(1 for m in harmful_response_markers if m in r)
        marker_score = min(marker_hits / 5.0, 1.0)
        
        # Check if response is answering an instructional harmful prompt
        if p_intent == "instructional" and (marker_score > 0.3 or len(r.split()) > 20):
            return min(0.5 + marker_score, 1.0)
        
        return marker_score * 0.6

    def _compute_layer_anomaly_score(self, hidden_states) -> float:
        """Detect anomaly in layer activations (early warning)."""
        if not hidden_states or len(hidden_states) < 2:
            return 0.0
        
        # Compare consecutive layer norms
        norms = [torch.norm(h).item() for h in hidden_states]
        if len(norms) < 3:
            return 0.0
        
        # Spike detection
        anomalies = []
        for i in range(1, len(norms) - 1):
            if norms[i] > norms[i-1] * 2.5 and norms[i] > norms[i+1] * 1.5:
                anomalies.append(norms[i])
        
        if anomalies:
            return min(0.3 + len(anomalies) * 0.1, 1.0)
        return 0.0

    def _find_responsible_layer(self, hidden_states) -> int:
        """Find which layer shows strongest anomaly."""
        if not hidden_states:
            return 0
        
        norms = [torch.norm(h).item() for h in hidden_states]
        if len(norms) < 2:
            return 0
        
        max_spike_idx = 0
        max_spike = 0
        for i in range(1, len(norms)):
            spike = norms[i] - norms[i-1]
            if spike > max_spike:
                max_spike = spike
                max_spike_idx = i
        
        return max_spike_idx
        
        return max_spike_idx
    def _response_is_providing_harm(self, decoded: str, intent_result: dict) -> bool:
        """
        Check whether the model's in-progress response is actually providing
        harmful content - not just mentioning a risky word.
 
        Looks for: step-by-step structure + harmful category words together.
        """
        categories = intent_result.get("categories", [])
        triggered_words = intent_result.get("triggered_words", [])
 
        # Instructional structure markers in the response
        instruction_response_markers = [
            "step 1", "step 2", "first you", "firstly,", "to begin,",
            "here's how", "here is how", "the process is", "you will need to",
        ]
 
        has_instruction_structure = any(m in decoded for m in instruction_response_markers)
        has_trigger_word = any(w in decoded for w in triggered_words)
 
        return has_instruction_structure and has_trigger_word
 
    # -------------------------------------------------------------------------
    # SAFETY ANALYSIS
    # -------------------------------------------------------------------------
 
    def _perform_safety_analysis(self, prompt, hidden_states):
        """
        Comprehensive safety analysis using intent-aware token analysis,
        layer-level analysis, and semantic checks.
        """
        token_analysis = self._analyze_tokens(prompt)
        layer_analysis = self._analyze_layers(hidden_states)
        semantic_signal = self._semantic_intent_check(prompt)
        intent_decision = self._decide_intent(token_analysis, layer_analysis, semantic_signal)
 
        return {
            "token_analysis": token_analysis,
            "layer_analysis": layer_analysis,
            "intent_decision": intent_decision,
            "semantic_signal": semantic_signal,
        }
 
    def _analyze_tokens(self, prompt: str) -> dict:
        """
        Intent-aware token analysis.
 
        FIXED: No longer scores words like 'dangerous' or 'kill' in purely
        informational or medical contexts. Uses IntentAnalyzer for context.
        """
        intent_result = self._intent_analyzer.analyze(prompt)
 
        tokens = self.tokenizer.tokenize(prompt)
 
        return {
            "tokens": tokens,
            "risk_score": intent_result["effective_risk"],
            "raw_risk_score": intent_result["raw_risk"],
            "categories": intent_result["categories"],
            "phrases": intent_result["triggered_words"],
            "intent_type": intent_result["framing"],
            "token_intent": intent_result["intent_label"],
            "safe_context_score": intent_result["safe_context_score"],
            "mitigation_applied": intent_result["raw_risk"] - intent_result["effective_risk"],
        }
 
    def _analyze_layers(self, hidden_states) -> dict:
        """Analyze hidden states for layer-level safety signals."""
        if not hidden_states or len(hidden_states) < 2:
            return {
                "num_layers": 0,
                "middle_layer_mean": 0.0,
                "final_layer_mean": 0.0,
                "cosine_similarity": 0.0,
                "layer_signal": "uncertain"
            }
 
        num_layers = len(hidden_states)
 
        middle_layer = hidden_states[num_layers // 2]
        final_layer = hidden_states[-1]
 
        middle_mean = torch.mean(torch.abs(middle_layer)).item()
        final_mean = torch.mean(torch.abs(final_layer)).item()
 
        middle_vec = middle_layer.mean(dim=1).flatten().detach().cpu().numpy().reshape(1, -1)
        final_vec = final_layer.mean(dim=1).flatten().detach().cpu().numpy().reshape(1, -1)
 
        if SKLEARN_AVAILABLE:
            cos_sim = cosine_similarity(middle_vec, final_vec)[0][0]
        else:
            dot_product = np.dot(middle_vec.flatten(), final_vec.flatten())
            norm_m = np.linalg.norm(middle_vec)
            norm_f = np.linalg.norm(final_vec)
            cos_sim = dot_product / (norm_m * norm_f) if norm_m * norm_f > 0 else 0.5
 
        if cos_sim < 0.4:
            layer_signal = "high"
        elif cos_sim > 0.75:
            layer_signal = "low"
        else:
            layer_signal = "uncertain"
 
        return {
            "num_layers": num_layers,
            "middle_layer_mean": middle_mean,
            "final_layer_mean": final_mean,
            "cosine_similarity": float(cos_sim),
            "layer_signal": layer_signal,
        }
 
    def _semantic_intent_check(self, prompt: str) -> str:
        """Check semantic complexity of prompt via embedding norm."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
 
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
 
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden.mean(dim=1)
        score = torch.norm(embedding).item()
 
        if score > 15:
            return "complex"
        elif score < 8:
            return "simple"
        else:
            return "moderate"
 
    def _decide_intent(self, token_result: dict, layer_result: dict, semantic_signal: str) -> dict:
        """
        Combine all analyses to make final safety decision.
 
        FIXED:
        - Uses effective_risk (intent-adjusted) not raw keyword hits
        - Informational + safe context → ALLOW even if risk words present
        - Only BLOCK when both token-level AND layer-level signals agree
        - MONITOR for ambiguous cases rather than false blocking
        """
        token_intent = token_result["token_intent"]
        effective_risk = token_result["risk_score"]
        framing = token_result["intent_type"]
        layer_signal = layer_result["layer_signal"]
        safe_ctx = token_result.get("safe_context_score", 0.0)
        categories = token_result.get("categories", [])
 
        # Absolute block: child safety or bioweapon regardless of context
        if "child_safety" in categories or "bioweapon" in categories:
            return {
                "final_decision": "BLOCK",
                "reason": "Absolute risk category detected",
                "confidence": 0.99,
            }
 
        # Strong harmful: high risk + instructional framing + layer confirmation
        if effective_risk >= 4.0 and framing == "instructional" and layer_signal == "high":
            return {
                "final_decision": "BLOCK",
                "reason": "High-risk instructional query with layer confirmation",
                "confidence": 0.95,
            }
 
        # Instructional + moderately risky (no safe context)
        if framing == "instructional" and effective_risk >= 3.0 and safe_ctx < 0.2:
            return {
                "final_decision": "BLOCK",
                "reason": "Instructional harmful query",
                "confidence": 0.88,
            }
 
        # Safe informational (e.g. "is cancer dangerous?", "what is hacking?")
        if token_intent in ("informational", "neutral") and effective_risk < 2.0:
            return {
                "final_decision": "ALLOW",
                "reason": "Low-risk informational query",
                "confidence": 0.90,
            }
 
        # Safe context strongly present (medical, academic, fictional)
        if safe_ctx >= 0.5 and effective_risk < 4.0:
            return {
                "final_decision": "ALLOW",
                "reason": f"Safe context mitigates risk (context score: {safe_ctx:.2f})",
                "confidence": 0.85,
            }
 
        # Ambiguous: layer signal elevated but intent unclear
        if layer_signal == "high" or semantic_signal == "complex":
            return {
                "final_decision": "MONITOR",
                "reason": "Ambiguous semantic complexity detected",
                "confidence": 0.65,
            }
 
        return {
            "final_decision": "ALLOW",
            "reason": "Low-risk query",
            "confidence": 0.80,
        }