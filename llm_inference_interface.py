"""
LLM Model Inference Interface
A Streamlit web application for running inference on multiple small LLM models.
Provides access to hidden states and attention weights for analysis and interventions.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# SAFETY ANALYSIS MODULES
# ============================================================================

# NEW: Step-by-step generation with monitoring
def generate_with_monitoring(model, tokenizer, input_ids, max_new_tokens=100):

    generated_ids = input_ids.clone()
    trigger_info = None

    for step in range(max_new_tokens):

        outputs = model(
            input_ids=generated_ids,
            output_hidden_states=True,
            return_dict=True
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

        token_str = tokenizer.decode(next_token[0])

        # 🔥 Trigger detection
        decoded_so_far = tokenizer.decode(generated_ids[0]).lower()
        if any(word in decoded_so_far for word in ["kill", "bomb", "attack"]):
            trigger_info = {
                "token": token_str,
                "position": generated_ids.shape[1]
            }
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return generated_ids, trigger_info, outputs.hidden_states
    
# NEW: Token-level analysis for intent detection
def tokenize_and_analyze(prompt: str, tokenizer) -> Dict[str, Any]:
    """
    Perform token-level analysis for intent detection.

    Args:
        prompt: User input prompt
        tokenizer: Model tokenizer

    Returns:
        dict: Token analysis results
    """
    # Harmful keywords database
    # NEW: Weighted keyword categories (not just flat list)
    RISK_PATTERNS = {
        "violence": ["kill", "attack", "weapon", "murder", "assault"],
        "illegal": ["hack", "breach", "exploit", "steal"],
        "dangerous": ["bomb", "explosive", "poison"],
        "self_harm": ["suicide", "harm myself"],
    }

    INTENT_PATTERNS = {
        "instruction": ["how to", "steps to", "guide to", "method to"],
        "curiosity": ["what is", "why", "explain", "is it"],
    }

    # Intent indicators
    INFORMATIONAL_WORDS = {"what", "is", "why", "how", "explain", "tell", "describe"}
    ACTION_WORDS = {"make", "build", "create", "do", "perform", "execute", "construct"}

    tokens = tokenizer.tokenize(prompt)
    prompt_lower = prompt.lower()

    risk_score = 0
    detected_categories = []
    detected_phrases = []

    # 🔥 Pattern-based detection (not just words)
    for category, words in RISK_PATTERNS.items():
        for word in words:
            if word in prompt_lower:
                risk_score += 2
                detected_categories.append(category)
                detected_phrases.append(word)

    # 🔥 Intent pattern detection
    intent_type = "neutral"

    for phrase in INTENT_PATTERNS["instruction"]:
        if phrase in prompt_lower:
            intent_type = "instruction"
            risk_score += 3

    for phrase in INTENT_PATTERNS["curiosity"]:
        if phrase in prompt_lower:
            if intent_type != "instruction":
                intent_type = "informational"

    # 🔥 Context-aware adjustment
    if "disease" in prompt_lower or "cancer" in prompt_lower:
        risk_score -= 2  # reduce false positives

    # 🔥 Final token-level intent
    if risk_score >= 5:
        token_intent = "harmful"
    elif intent_type == "informational":
        token_intent = "informational"
    else:
        token_intent = "neutral"

    return {
        "tokens": tokens,
        "risk_score": risk_score,
        "categories": list(set(detected_categories)),
        "phrases": detected_phrases,
        "intent_type": intent_type,
        "token_intent": token_intent
    }
def semantic_intent_check(prompt, tokenizer, model):

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]
    embedding = last_hidden.mean(dim=1)

    score = torch.norm(embedding).item()

    if score > 15:
        return "complex"
    elif score < 8:
        return "simple"
    else:
        return "moderate"
# NEW: Layer-level analysis
def analyze_layers(hidden_states):

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

    cos_sim = cosine_similarity(middle_vec, final_vec)[0][0]

    # 🔥 Improved logic
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
        "cosine_similarity": cos_sim,
        "layer_signal": layer_signal
    }
# NEW: Intent decision engine combining token and layer analysis
def decide_intent(token_result, layer_result, semantic_signal):

    token_intent = token_result["token_intent"]
    risk_score = token_result["risk_score"]
    layer_signal = layer_result["layer_signal"]

    # 🔥 Strong harmful case
    if risk_score >= 6 and layer_signal == "high":
        return {"final_decision": "BLOCK", "reason": "High-risk multi-signal detection", "confidence": 0.95}

    # 🔥 Instruction-based risky
    if token_result["intent_type"] == "instruction" and risk_score >= 4:
        return {"final_decision": "BLOCK", "reason": "Instructional harmful query", "confidence": 0.9}

    # 🔥 Informational safe
    if token_intent == "informational" and layer_signal == "low":
        return {"final_decision": "ALLOW", "reason": "Safe informational intent", "confidence": 0.85}

    # 🔥 Suspicious but unclear
    if layer_signal == "high" or semantic_signal == "complex":
        return {"final_decision": "MONITOR", "reason": "Semantic ambiguity detected", "confidence": 0.7}

    return {"final_decision": "ALLOW", "reason": "Low-risk query", "confidence": 0.8}

# NEW: Controlled response generation based on safety decision
def generate_controlled_response(
    decision: str,
    model,
    tokenizer,
    inputs: Dict,
    generation_config: Dict,
    prompt: str
) -> str:
    """
    Generate response based on safety decision.

    Args:
        decision: Safety decision ("ALLOW", "BLOCK", "MONITOR")
        model: The language model
        tokenizer: Model tokenizer
        inputs: Tokenized inputs
        generation_config: Generation parameters
        prompt: Original prompt

    Returns:
        str: Controlled response
    """
    if decision == "BLOCK":
        return "⚠️ **SAFETY ALERT**: This request contains potentially harmful content. I cannot provide assistance with this type of query. Please ask about safe, constructive topics."

    elif decision == "MONITOR":
        # Generate response but add disclaimer
        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **generation_config)

        input_length = inputs.input_ids.shape[1]
        generated_text = tokenizer.decode(
            generated_outputs[0][input_length:],
            skip_special_tokens=True
        )

        disclaimer = "⚠️ **MONITORED RESPONSE**: This query triggered safety monitoring. "
        return disclaimer + generated_text

    else:  # ALLOW
        # Generate normal response
        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **generation_config)

        input_length = inputs.input_ids.shape[1]
        generated_text = tokenizer.decode(
            generated_outputs[0][input_length:],
            skip_special_tokens=True
        )

        return generated_text, generated_outputs

# NEW: Logging hook for safety analysis results
def log_safety_analysis(safety_results: Dict, prompt: str, model_key: str) -> None:
    """
    Log safety analysis results for monitoring and improvement.
    Placeholder for future logging implementation.

    Args:
        safety_results: Complete safety analysis results
        prompt: Original user prompt
        model_key: Model used for analysis
    """
    # Placeholder: Could log to file, database, or external service
    # Example: Save to JSON file, send to monitoring service, etc.
    pass

# NEW: Hook for plugging in trained classifier
def classifier_prediction_placeholder(
    token_features: Dict,
    layer_features: Dict,
    model_key: str
) -> Optional[float]:
    """
    Placeholder for trained classifier prediction.
    Return confidence score for harmful intent (0-1).

    Args:
        token_features: Token-level analysis results
        layer_features: Layer-level analysis results
        model_key: Model identifier

    Returns:
        float or None: Classifier confidence score
    """
    # Placeholder: Load and use trained model here
    # Example: Use sklearn model, PyTorch model, etc.
    return None  # Return None to use rule-based system

# ============================================================================
# MODEL MANAGER CLASS
# ============================================================================
class ModelManager:
    """
    Manages loading and inference for multiple LLM models.
    Provides access to hidden states and attention weights.
    """

    def __init__(self):
        self.models = {}  # Cache for loaded models
        self.tokenizers = {}  # Cache for tokenizers
        self.device = "cpu"

        # Model configurations
        self.model_configs = {
            "TinyLlama (1.1B)": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_length": 512
            },
            "microsoft/phi-2": {
                "name": "microsoft/phi-2",
                "max_length": 1024
            },
            "distilgpt2": {
                "name": "distilgpt2",
                "max_length": 512
            }
        }

    @st.cache_resource
    def load_model(_self, model_key: str):
        """
        Load and cache a model and its tokenizer.

        Args:
            model_key: Key from model_configs

        Returns:
            tuple: (model, tokenizer)
        """
        if model_key not in _self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")

        config = _self.model_configs[model_key]

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config["name"],
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()

        return model, tokenizer

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """
        Get basic information about a model.

        Args:
            model_key: Key from model_configs

        Returns:
            dict: Model information
        """
        if model_key not in self.models:
            self.models[model_key], self.tokenizers[model_key] = self.load_model(model_key)

        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]

        # Get number of layers
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                num_layers = model.config.num_hidden_layers
            elif hasattr(model.config, 'n_layer'):
                num_layers = model.config.n_layer
            else:
                num_layers = len(model.model.layers) if hasattr(model, 'model') else 0
        else:
            num_layers = 0

        return {
            "num_layers": num_layers,
            "vocab_size": len(tokenizer) if hasattr(tokenizer, '__len__') else "Unknown",
            "max_position_embeddings": getattr(model.config, 'max_position_embeddings', 'Unknown')
        }

    def run_inference(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        intervention: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run inference on a model and capture internal states.

        Args:
            model_key: Model to use
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            intervention: Optional intervention config (for future use)

        Returns:
            dict: Inference results including generated text and internal states
        """
        if model_key not in self.models:
            self.models[model_key], self.tokenizers[model_key] = self.load_model(model_key)

        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]

        # NEW: Step 1 - Token-level analysis
        token_analysis = tokenize_and_analyze(prompt, tokenizer)

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # First, get hidden states and attentions from a forward pass
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

        # Extract information
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

        # NEW: Step 2 - Layer-level analysis
        layer_analysis = analyze_layers(hidden_states)
        semantic_signal = semantic_intent_check(prompt, tokenizer, model)
        # NEW: Step 3 - Intent decision (with classifier hook)
        # Try classifier first, fall back to rule-based
        classifier_confidence = classifier_prediction_placeholder(
            token_analysis, layer_analysis, model_key
        )

        if classifier_confidence is not None:
            # Use classifier result
            intent_decision = {
                "final_decision": "BLOCK" if classifier_confidence > 0.7 else "ALLOW",
                "reason": f"Classifier confidence: {classifier_confidence:.2f}",
                "confidence": classifier_confidence
            }
        else:
            # Use rule-based system
            intent_decision = decide_intent(token_analysis, layer_analysis, semantic_signal)

        # NEW: Log safety analysis
        log_safety_analysis({
            "token_analysis": token_analysis,
            "layer_analysis": layer_analysis,
            "intent_decision": intent_decision,
            "model_key": model_key,
            "timestamp": time.time()
        }, prompt, model_key)

        # NEW: Apply safety-based hidden state modifications
        modified_hidden_states = self.modify_hidden_states_for_safety(
            hidden_states, intent_decision["final_decision"], model_key
        )

        # NEW: Step 4 - Controlled response generation
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1
        }

        # NEW: Controlled step-by-step generation
        generated_ids, trigger_info, hidden_states = generate_with_monitoring(
            model,
            tokenizer,
            inputs.input_ids,
            max_new_tokens
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if trigger_info:
            # Apply intervention on detected harmful case
            hidden_states = self.modify_hidden_states_for_safety(
                hidden_states,
                "BLOCK",
                model_key
            )
        # For blocked responses, we don't have generated outputs
        if intent_decision["final_decision"] == "BLOCK":
            generated_outputs = inputs.input_ids  # Just use input for length calculation

        # Decode generated text (skip if blocked)
        input_length = inputs.input_ids.shape[1]
        if intent_decision["final_decision"] == "BLOCK":
            generated_length = 0
        else:
            generated_length = len(generated_outputs[0]) - input_length

        # Prepare analysis data
        analysis = {
            "num_layers": len(hidden_states) if hidden_states else 0,
            "hidden_states_shapes": [
                tuple(hs.shape) for hs in hidden_states
            ] if hidden_states else [],
            "attention_shapes": [
                tuple(att.shape) for att in attentions
            ] if attentions else [],
            "input_length": input_length,
            "generated_length": generated_length
        }

        # NEW: Add safety analysis results
        safety_results = {
            "token_analysis": token_analysis,
            "layer_analysis": layer_analysis,
            "intent_decision": intent_decision,
            "semantic_signal": semantic_signal   # NEW
        }

        return {
            "generated_text": generated_text,
            "analysis": analysis,
            "safety_results": safety_results,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "trigger_info": trigger_info   # NEW
        }

    def apply_intervention_placeholder(
        self,
        hidden_states: List[torch.Tensor],
        intervention_config: Dict
    ) -> List[torch.Tensor]:
        """
        Placeholder for applying interventions to hidden states.
        Modify this function to implement different intervention types.

        Args:
            hidden_states: List of hidden state tensors
            intervention_config: Configuration for intervention

        Returns:
            List[torch.Tensor]: Modified hidden states
        """
        # Example interventions (not implemented yet):
        # - Zero out specific layers
        # - Add noise to hidden states
        # - Scale hidden states
        # - Apply safety filters

        intervention_type = intervention_config.get("type", "none")

        if intervention_type == "none":
            return hidden_states
        elif intervention_type == "zero_layer":
            layer_idx = intervention_config.get("layer_idx", 0)
            if layer_idx < len(hidden_states):
                modified_states = hidden_states.copy()
                modified_states[layer_idx] = torch.zeros_like(hidden_states[layer_idx])
                return modified_states
        elif intervention_type == "add_noise":
            noise_std = intervention_config.get("noise_std", 0.1)
            return [hs + torch.randn_like(hs) * noise_std for hs in hidden_states]
        elif intervention_type == "safety_filter":
            # NEW: Placeholder for safety-based interventions
            # Could modify hidden states to reduce harmful outputs
            return hidden_states

        # Default: return unchanged
        return hidden_states

    # NEW: Hook for pre-generation hidden state modification
    def modify_hidden_states_for_safety(
        self,
        hidden_states: List[torch.Tensor],
        safety_decision: str,
        model_key: str
    ) -> List[torch.Tensor]:
        """
        Modify hidden states before generation based on safety decision.
        Placeholder for advanced safety interventions.

        Args:
            hidden_states: Original hidden states
            safety_decision: Safety decision ("ALLOW", "BLOCK", "MONITOR")
            model_key: Model identifier

        Returns:
            List[torch.Tensor]: Modified hidden states
        """
        if safety_decision == "BLOCK":
            return [hs * 0.0 for hs in hidden_states]

        elif safety_decision == "MONITOR":
            return [hs * 0.7 for hs in hidden_states]
            # ALLOW: return unchanged
        return hidden_states

# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="LLM Inference Interface",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🛡️ Safe LLM Inference Interface")
    st.markdown("Advanced intent-aware safety system for LLM misbehavior detection with token and layer analysis.")

    # Initialize session state
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "TinyLlama (1.1B)"
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        model_options = list(st.session_state.model_manager.model_configs.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=model_options.index(st.session_state.current_model)
        )
        st.session_state.current_model = selected_model

        # Generation parameters
        st.subheader("Generation Parameters")
        max_tokens = st.slider("Max New Tokens", 50, 200, 100)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)

        # Model info
        if st.button("📊 Show Model Info"):
            try:
                info = st.session_state.model_manager.get_model_info(selected_model)
                st.json(info)
            except Exception as e:
                st.error(f"Error loading model info: {e}")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Input Prompt")

        # Prompt input
        prompt = st.text_area(
            "Enter your prompt:",
            value="How to hack a website?",
            height=100,
            help="Enter the text prompt for the model to continue or respond to."
        )

        # Generate button
        if st.button("🚀 Generate Output", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("Please enter a prompt.")
            else:
                with st.spinner(f"Running inference on {selected_model}..."):
                    start_time = time.time()

                    try:
                        result = st.session_state.model_manager.run_inference(
                            selected_model,
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )

                        st.session_state.last_result = result
                        st.session_state.last_result['inference_time'] = time.time() - start_time

                        st.success(f"✅ Inference completed in {st.session_state.last_result['inference_time']:.2f}s")

                    except Exception as e:
                        st.error(f"Error during inference: {e}")
                        st.session_state.last_result = None

        # Display results
        if st.session_state.last_result:
            st.subheader("📄 Generated Output")

            # Show the full prompt + generated text
            full_output = prompt + st.session_state.last_result['generated_text']
            st.text_area(
                "Complete Output:",
                value=full_output,
                height=200,
                disabled=True
            )

            # Show just the generated part
            st.info(f"**Generated Text:** {st.session_state.last_result['generated_text']}")
            trigger = st.session_state.last_result.get("trigger_info", None)

        if trigger:
            st.error(f"⚠️ Trigger Token: {trigger['token']}")
            st.write(f"Position: {trigger['position']}")

    with col2:
        st.subheader("🔍 Analysis")

        if st.session_state.last_result:
            analysis = st.session_state.last_result['analysis']

            # Basic metrics
            st.metric("Inference Time", f"{st.session_state.last_result['inference_time']:.2f}s")
            st.metric("Input Length", analysis['input_length'])
            st.metric("Generated Length", analysis['generated_length'])
            st.metric("Model Layers", analysis['num_layers'])

            # NEW: Safety Analysis Section
            if 'safety_results' in st.session_state.last_result:
                safety = st.session_state.last_result['safety_results']

                st.markdown("---")
                st.subheader("🛡️ Safety Analysis")

                # Token Analysis
                with st.expander("📝 Token-Level Analysis", expanded=True):
                    token_analysis = safety['token_analysis']
                    st.write(f"**Intent:** {token_analysis['token_intent'].upper()}")
                    st.write(f"**Tokens:** {len(token_analysis['tokens'])}")
                    st.write(f"**Risk Score:** {token_analysis['risk_score']}")
                    st.write(f"**Categories:** {token_analysis['categories']}")
                    st.write(f"**Intent Type:** {token_analysis['intent_type']}")

                # Layer Analysis
                with st.expander("🏗️ Layer-Level Analysis", expanded=True):
                    layer_analysis = safety['layer_analysis']
                    if layer_analysis["layer_signal"] == "high":
                        st.warning("⚠️ High-risk layer activity detected")
                    st.write(f"**Signal:** {layer_analysis['layer_signal'].upper()}")
                    st.metric("Middle Layer Mean", f"{layer_analysis['middle_layer_mean']:.4f}")
                    st.metric("Final Layer Mean", f"{layer_analysis['final_layer_mean']:.4f}")
                    st.metric("Cosine Similarity", f"{layer_analysis['cosine_similarity']:.4f}")
                    semantic_signal = safety['semantic_signal']
                    st.write(f"**Semantic Signal:** {semantic_signal}")
                    suggestion = "zero" if layer_analysis["layer_signal"] == "high" else "scale"
                    st.info(f"💡 Suggested Intervention: {suggestion}")
                # Intent Decision
                with st.expander("🎯 Intent Decision", expanded=True):
                    decision = safety['intent_decision']
                    if decision['final_decision'] == "BLOCK":
                        st.error("🚫 Harmful intent detected")
                    elif decision['final_decision'] == "MONITOR":
                        st.warning("⚠️ Suspicious intent detected")
                    else:
                        st.success("✅ Safe query")
                    decision_color = {
                        "ALLOW": "🟢",
                        "BLOCK": "🔴",
                        "MONITOR": "🟡"
                    }.get(decision['final_decision'], "⚪")

                    st.write(f"**Decision:** {decision_color} {decision['final_decision']}")
                    st.write(f"**Reason:** {decision['reason']}")
                    # 🔥 ADD HERE
                    st.progress(decision['confidence'])
                    st.write(f"Confidence: {decision['confidence']*100:.1f}%")

            # Hidden states info
            if analysis['hidden_states_shapes']:
                st.subheader("Hidden States")
                for i, shape in enumerate(analysis['hidden_states_shapes']):
                    st.write(f"Layer {i}: {shape}")

            # Attention info
            if analysis['attention_shapes']:
                st.subheader("Attention Weights")
                st.write(f"Available: {len(analysis['attention_shapes'])} layers")
                for i, shape in enumerate(analysis['attention_shapes'][:3]):  # Show first 3
                    st.write(f"Layer {i}: {shape}")
                if len(analysis['attention_shapes']) > 3:
                    st.write(f"... and {len(analysis['attention_shapes']) - 3} more layers")

        else:
            st.info("Run inference to see analysis.")

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit and HuggingFace Transformers")
    st.caption("Models run on CPU for accessibility")

if __name__ == "__main__":
    main()