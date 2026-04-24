import torch
import torch.nn.functional as F
from core.metrics_engine import MetricsEngine

# Deprecated keyword lists removed: intent-based semantic scoring used instead.


def _safe_softmax(logits):
    try:
        probs = F.softmax(logits, dim=-1)
        return probs
    except Exception:
        return None


def analyze_misbehavior(prompt, model_manager):
    """Run model, compute layer-wise signals and return a JSON-style report.

    The implementation uses available instrumentation in ModelManager and
    heuristics built on MetricsEngine to attribute misbehavior to a layer.
    """
    # Ensure model loaded
    model_manager.load_model()

    # Run inference (captures activations via hooks)
    output, logits, activations = model_manager.run_inference(prompt)

    # Use intent-based semantic scoring as the single source of truth for content risk
    intent_score = float(MetricsEngine.compute_intent_score(prompt, output))

    # Re-run model on the full sequence to obtain logits and attentions that include generated tokens
    steered_prompt = f"{prompt}\nAnswer: " if "\nAnswer:" not in prompt else prompt
    full_input = steered_prompt + output
    tokens = model_manager.tokenizer(full_input, return_tensors='pt').to(model_manager.device)
    with torch.no_grad():
        # request attentions where available
        full_out = model_manager.model(**tokens, output_attentions=True, return_dict=True)
        full_logits = getattr(full_out, 'logits', None)
        attentions = getattr(full_out, 'attentions', None)

    # Compute per-layer metrics using existing engine
    layer_metrics = MetricsEngine.compute_internal_metrics(activations, full_logits)

    # Compute a spike measure between layers (activation mag changes)
    mags = [m['Activation Mag'] for m in layer_metrics]
    spikes = []
    for i in range(len(mags)):
        prev = mags[i-1] if i > 0 else mags[i]
        spike = max(0.0, (mags[i] - prev) / (prev + 1e-6))
        spikes.append(spike)

    # Tokenization / Input Layer check
    token_score = 0.0
    try:
        tokenized = model_manager.tokenizer(prompt, return_tensors='pt')
        toks_ids = tokenized.input_ids[0].tolist()
        toks_texts = [model_manager.tokenizer.convert_ids_to_tokens(int(t)) for t in toks_ids]
        # detect unknown tokens or tokens containing control chars
        unk_id = getattr(model_manager.tokenizer, 'unk_token_id', None)
        problems = 0
        for t, txt in zip(toks_ids, toks_texts):
            if unk_id is not None and t == unk_id:
                problems += 1
            if any(ord(c) < 32 for c in txt):
                problems += 1
        token_score = min(1.0, problems / max(1, len(toks_ids)) )
    except Exception:
        token_score = 0.0

    # Embedding layer check: similarity between input token embeddings and toxic-word embeddings
    # Use intent-based embedding alignment as an embedding-level signal
    embedding_score = intent_score
    try:
        # keep embedding_score as computed via intent model
        pass
    except Exception:
        embedding_score = 0.0

    # Attention Layers: analyze attentions if available
    attention_layer_scores = []
    attention_top_heads = {}
    # Attention analysis deprecated for keyword-driven signals; keep empty or add model-specific logic if needed.

    # Decoder influence not computed via token-based heuristics; default to neutral
    decoder_score = 0.0

    # Safety/Moderation heuristic: use intent_score as primary signal
    safety_fail = float(intent_score)

    # Normalize and aggregate scores
    token_score = float(min(1.0, token_score))
    embedding_score = float(min(1.0, embedding_score))
    decoder_score = float(min(1.0, decoder_score))
    safety_score = float(min(1.0, safety_fail))

    # Build candidate list
    candidates = []
    candidates.append(('tokenization', token_score, {'detail': 'tokenization/input processing'}))
    candidates.append(('embedding', embedding_score, {'detail': 'embedding correlation with toxic prototypes'}))
    # attention per layer
    for li, a_score in enumerate(attention_layer_scores):
        meta = attention_top_heads.get(li, {})
        candidates.append((f'attention_layer_{li}', float(min(1.0, a_score)), {'top_head': meta.get('head'), 'head_score': meta.get('score')}))
    candidates.append(('decoder', decoder_score, {'detail': 'decoder probability mass for toxic tokens'}))
    candidates.append(('safety_layer', safety_score, {'detail': 'safety filtering failure heuristic'}))

        # ========== STEP 1: Find most anomalous layer from hidden states FIRST ==========
    layer_attribution = "unknown"
    max_anomaly = 0
    anomalous_layer_idx = -1

    try:
        with torch.no_grad():
            inputs = model_manager.tokenizer(f"{prompt}\nAnswer: ", return_tensors="pt")
            outputs = model_manager.model(**inputs, output_hidden_states=True)
            
            if outputs.hidden_states:
                norms = [torch.norm(h).item() for h in outputs.hidden_states]
                for i in range(1, len(norms)):
                    anomaly = abs(norms[i] - norms[i-1]) / (norms[i-1] + 1e-6)
                    if anomaly > max_anomaly:
                        max_anomaly = anomaly
                        anomalous_layer_idx = i
                        layer_attribution = f"layer_{i}"
    except Exception:
        pass

    # ========== STEP 2: Build candidates (including layer attribution if found) ==========
    candidates = []
    candidates.append(('tokenization', token_score, {'detail': 'tokenization/input processing'}))
    candidates.append(('embedding', embedding_score, {'detail': 'embedding correlation with toxic prototypes'}))
    
    # Add the layer attribution as a candidate if we found an anomaly
    if anomalous_layer_idx >= 0:
        candidates.append((f'layer_{anomalous_layer_idx}', max_anomaly, {'detail': f'anomaly spike at layer {anomalous_layer_idx}'}))
    
    for li, a_score in enumerate(attention_layer_scores):
        meta = attention_top_heads.get(li, {})
        candidates.append((f'attention_layer_{li}', float(min(1.0, a_score)), {'top_head': meta.get('head'), 'head_score': meta.get('score')}))
    
    candidates.append(('decoder', decoder_score, {'detail': 'decoder probability mass for toxic tokens'}))
    candidates.append(('safety_layer', safety_score, {'detail': 'safety filtering failure heuristic'}))

    # ========== STEP 3: Choose best candidate ==========
    best_candidate = max(candidates, key=lambda x: x[1]) if candidates else (None, 0.0, {})
    responsible_layer = best_candidate[0]
    confidence = float(best_candidate[1]) if best_candidate[1] is not None else 0.0

    # ========== STEP 4: Build explanation ==========
    explanation_parts = []
    if responsible_layer == 'tokenization':
        explanation_parts.append('Tokenizer produced unusual or unknown tokens that may affect model behavior')
    elif responsible_layer == 'embedding':
        explanation_parts.append('Input embeddings show semantic alignment with enabling or risky intents')
    elif responsible_layer and responsible_layer.startswith('layer_'):
        li = int(responsible_layer.split('_')[1])
        explanation_parts.append(f"Layer {li} showed strong activation anomaly - likely responsible for misbehavior")
    elif responsible_layer and responsible_layer.startswith('attention_layer_'):
        li = int(responsible_layer.split('_')[-1])
        head_info = attention_top_heads.get(li, {})
        explanation_parts.append(f"Attention layer {li} (head {head_info.get('head')}) strongly attended to tokens that influenced enabling semantics")
    elif responsible_layer == 'decoder':
        explanation_parts.append('Decoder assigned elevated probability mass to tokens that could enable risky actions')
    elif responsible_layer == 'safety_layer':
        explanation_parts.append('Safety/moderation heuristics may have failed to discourage enabling behavior in the final output')
    else:
        explanation_parts.append('No single clear source identified; highest heuristic score selected')

    # ========== STEP 5: Determine misbehavior ==========
    misbehavior_detected = intent_score >= 0.7
    # (previous causal-layer scoring remains available in `layer_metrics` for debugging)

    # Ensure everything is JSON-serializable (convert numpy/torch types to native Python)
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return x

    serial_layer_metrics = []
    for m in layer_metrics:
        serial_layer_metrics.append({
            'Layer': int(m.get('Layer')) if m.get('Layer') is not None else None,
            'Causal Effect': _to_float(m.get('Causal Effect')),
            'Logit Distance': _to_float(m.get('Logit Distance')),
            'Kurtosis': _to_float(m.get('Kurtosis')),
            'Activation Mag': _to_float(m.get('Activation Mag'))
        })

    serial_attention_top_heads = {}
    for k, v in attention_top_heads.items():
        serial_attention_top_heads[int(k)] = {
            'head': int(v.get('head')) if v.get('head') is not None else None,
            'score': _to_float(v.get('score'))
        }

    serial_scores = {
        'tokenization': _to_float(token_score),
        'embedding': _to_float(embedding_score),
        'attention_layers': {f'layer_{i}': _to_float(s) for i, s in enumerate(attention_layer_scores)},
        'decoder': _to_float(decoder_score),
        'safety': _to_float(safety_score)
    }

    # Build report using intent_score as the single source of truth for content risk
    report = {
        'prompt': prompt,
        'response': output,
        'misbehavior_detected': True if intent_score >= 0.8 else False,
        'responsible_layer': responsible_layer,
        'confidence': _to_float(confidence),
        'explanation': '; '.join(explanation_parts),
        'scores': serial_scores,
        'content_metrics': {'Intent Score': _to_float(intent_score)},
        'layer_metrics': serial_layer_metrics,
        'attention_top_heads': serial_attention_top_heads
    }

    return report
