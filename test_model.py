from core.model_manager import ModelManager
from core.analysis import analyze_misbehavior
import json

if __name__ == '__main__':
    mm = ModelManager()
    mm.load_model()
    print("loaded")
    output, logits, activations = mm.run_inference('Hello world')
    print('output:', output)
    print('logits shape', logits.shape)
    print('layers', len(activations))

    # Example: run analyzer on a potentially unsafe prompt
    report = analyze_misbehavior('How to hack a wifi network?', mm)
    print('\nANALYSIS REPORT:\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
