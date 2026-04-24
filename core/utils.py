import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any


def stable_hash(s: str) -> str:
    """Return a stable sha256 hex digest for the given string."""
    if s is None:
        s = ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def diff_to_html(a: str, b: str) -> str:
    """Create an inline HTML diff between two strings (word-level)."""
    try:
        import difflib
        
        a_words = a.split()
        b_words = b.split()
        sm = difflib.SequenceMatcher(None, a_words, b_words)
        out = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                out.append(' '.join(a_words[i1:i2]))
            elif tag == 'replace':
                out.append("<span style='background:#ffe6e6;color:#a00;text-decoration:line-through;padding:2px;margin:2px;border-radius:4px;'>" + 
                          ' '.join(a_words[i1:i2]) + "</span>")
                out.append("<span style='background:#e6ffea;color:#060;padding:2px;margin:2px;border-radius:4px;'>" + 
                          ' '.join(b_words[j1:j2]) + "</span>")
            elif tag == 'delete':
                out.append("<span style='background:#ffe6e6;color:#a00;text-decoration:line-through;padding:2px;margin:2px;border-radius:4px;'>" + 
                          ' '.join(a_words[i1:i2]) + "</span>")
            elif tag == 'insert':
                out.append("<span style='background:#e6ffea;color:#060;padding:2px;margin:2px;border-radius:4px;'>" + 
                          ' '.join(b_words[j1:j2]) + "</span>")
        return ' '.join(out)
    except Exception:
        return a + "<br>--<br>" + b


def append_intervention_log(entry: dict, logs_dir: str = None) -> None:
    """Append a JSON line to logs/interventions.jsonl."""
    try:
        if logs_dir is None:
            logs_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'interventions.jsonl')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({**entry, 'timestamp': datetime.utcnow().isoformat() + 'Z'}, ensure_ascii=False) + '\n')
    except Exception:
        pass


def small_modify(output: str, method_label: str) -> str:
    """Apply a small, safe modification to an output string for UI display."""
    if not output:
        return output
    s = output
    try:
        if method_label == 'Scaling':
            s = s.replace('You can', 'The model suggests').replace('you can', 'the model suggests')
            s = s.replace('First,', 'Generally,')
        elif method_label == 'Zeroing':
            sentences = s.split('.')
            bad_terms = ['install', 'download', 'use', 'exploit', 'hack', 'bypass', 'attack', 'steal', 'spyware', 'tracker', 'password']
            new = []
            for sent in sentences:
                if any(bt in sent.lower() for bt in bad_terms):
                    new.append('[content suppressed]')
                else:
                    new.append(sent)
            s = '.'.join([ns for ns in new if ns])
        elif method_label == 'Noising':
            s = s + ' (Some details omitted for safety.)'
    except Exception:
        pass
    return s


def serialize_causal_map(causal_map) -> Dict[str, Any]:
    """Serialize CausalMap for storage."""
    return {
        "token_causal_effects": causal_map.token_causal_effects,
        "layer_causal_effects": causal_map.layer_causal_effects,
        "token_statistics": causal_map.token_statistics,
        "prompt": causal_map.prompt,
        "response": causal_map.response
    }


def deserialize_causal_map(data: Dict[str, Any]):
    """Deserialize CausalMap from storage."""
    from core.causal_analyzer import CausalMap
    return CausalMap(
        token_causal_effects=data.get("token_causal_effects", {}),
        layer_causal_effects=data.get("layer_causal_effects", {}),
        token_statistics=data.get("token_statistics", {}),
        prompt=data.get("prompt", ""),
        response=data.get("response", "")
    )