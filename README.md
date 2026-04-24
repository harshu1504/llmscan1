# LLMSCAN: Causal Scan for LLM Misbehavior Detection

LLMSCAN is a tool for detecting and analyzing LLM misbehavior through causal scanning of internal layers.

## Features
- **Prompt Analysis**: Detect toxicity, bias, and hallucination in model outputs.
- **Internal Scanning**: Captures layer-wise activations and computes causal effects.
- **Intervention**: Modify activations in real-time (Zero, Scale, Noise) to observe changes in behavior.
- **Visualization**: Interactive charts for model-level metrics.
- **Safety Scoring**: Overall risk assessment and alerts.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- **Frontend/Backend**: Streamlit
- **Model**: TinyLlama-1.1B (via Hugging Face Transformers)
- **Deep Learning**: PyTorch
- **Visualization**: Plotly
