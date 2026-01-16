# Intent Classification

A simple and extensible Python library for classifying text into predefined intents using machine learning, featuring both traditional ML and neural multi-head attention approaches.

## Features

- **Two Classification Approaches**:
  - Traditional ML (TF-IDF + sklearn classifiers)
  - Neural Multi-Head Attention (transformer-based, inspired by UIAN architecture)
- **Text Preprocessing**: Built-in text preprocessing with customizable options
- **Confidence Scores**: Get confidence scores for predictions
- **Explainability**: Attention visualization for understanding model decisions
- **Risk Tier Classification**: UIAN-style 4-tier intent classification
- **Model Persistence**: Save and load trained models
- **CLI Interface**: Command-line tool for training and prediction

## Installation

Basic installation (traditional classifiers only):

```bash
pip install -e .
```

With neural classifier support:

```bash
pip install -e ".[neural]"
```

Full installation (including dev dependencies):

```bash
pip install -e ".[all]"
```

## Quick Start

### Traditional Classifier (IntentClassifier)

```python
from intent_classifier import IntentClassifier

# Prepare training data
texts = [
    "hello", "hi there", "hey",
    "goodbye", "bye", "see you later",
    "help me", "I need help", "assist me"
]
intents = [
    "greeting", "greeting", "greeting",
    "goodbye", "goodbye", "goodbye",
    "help", "help", "help"
]

# Train the classifier
classifier = IntentClassifier()
classifier.fit(texts, intents)

# Make predictions
result = classifier.predict("hello there")
print(result)  # "greeting"

# Get confidence scores
result = classifier.predict_with_confidence("hello there")
print(result)
# {"intent": "greeting", "confidence": 0.85, "all_scores": {...}}

# Save the model
classifier.save("model.pkl")
```

### Neural Multi-Head Attention Classifier

The `MultiHeadAttentionClassifier` uses a transformer-based architecture with multiple specialized attention heads, inspired by the UIAN (User Intent Attention Network) concept.

```python
from intent_classifier import MultiHeadAttentionClassifier

# Train the classifier
classifier = MultiHeadAttentionClassifier(
    model_name="distilbert-base-uncased",
    num_attention_heads=3,
)
classifier.fit(texts, intents, epochs=3)

# Make predictions with attention weights
result = classifier.predict_with_attention("hello there!")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2%}")

# Inspect which tokens the model attended to
for head in result['attention_weights']:
    print(f"\n{head['head_name']} head top tokens:")
    for token, weight in head['top_tokens']:
        print(f"  {token}: {weight:.4f}")
```

### Risk Tier Classification (UIAN-style)

Classify text into 4 risk tiers based on confidence scores:

```python
result = classifier.classify_risk_tier("suspicious text here")
print(f"Tier: {result['tier']} ({result['tier_name']})")
print(f"Description: {result['description']}")

# Tiers:
# - Tier 1 (Orchestrator): score > 0.8 - High confidence
# - Tier 2 (Committed): 0.5 < score <= 0.8 - Moderate-high confidence
# - Tier 3 (Casual): 0.2 < score <= 0.5 - Moderate confidence
# - Tier 4 (Low Risk): score <= 0.2 - Low confidence
```

### Attention Visualization

Visualize which parts of the input text the model focuses on:

```python
# Text format (for terminal)
viz = classifier.visualize_attention("hello there", head_index=0)
print(viz)

# HTML format (for notebooks/web)
html_viz = classifier.visualize_attention("hello there", output_format="html")
```

## Command Line Interface

Train a model:

```bash
intent-classify train -d data/sample_intents.json -o model.pkl
```

Predict intents:

```bash
# Single prediction
intent-classify predict -m model.pkl "hello there"

# Interactive mode
intent-classify predict -m model.pkl
```

Show model information:

```bash
intent-classify info -m model.pkl
```

## Training Data Format

Training data should be in JSON format:

```json
{
  "intents": [
    {
      "name": "greeting",
      "examples": ["hello", "hi there", "hey"]
    },
    {
      "name": "goodbye",
      "examples": ["bye", "goodbye", "see you"]
    }
  ]
}
```

## Configuration Options

### IntentClassifier Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | str | "logistic_regression" | Classification algorithm ("logistic_regression" or "naive_bayes") |
| `preprocessor` | TextPreprocessor | None | Custom text preprocessor |
| `confidence_threshold` | float | 0.0 | Minimum confidence for predictions |

### MultiHeadAttentionClassifier Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "distilbert-base-uncased" | Pre-trained transformer model |
| `num_attention_heads` | int | 3 | Number of specialized attention heads |
| `attention_head_configs` | list | DEFAULT_ATTENTION_HEADS | Custom head configurations |
| `hidden_dim` | int | 256 | Hidden layer dimension |
| `dropout` | float | 0.1 | Dropout probability |
| `max_length` | int | 128 | Maximum sequence length |
| `confidence_threshold` | float | 0.0 | Minimum confidence for predictions |
| `device` | str | None | Device ('cuda', 'cpu', or auto-detect) |

### Custom Attention Heads

Define custom attention heads for specialized analysis:

```python
from intent_classifier import AttentionHead, MultiHeadAttentionClassifier

custom_heads = [
    AttentionHead(
        name="sentiment",
        description="Analyzes emotional tone and sentiment"
    ),
    AttentionHead(
        name="urgency",
        description="Detects urgency markers in text"
    ),
]

classifier = MultiHeadAttentionClassifier(
    num_attention_heads=2,
    attention_head_configs=custom_heads
)
```

### TextPreprocessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowercase` | bool | True | Convert text to lowercase |
| `remove_punctuation` | bool | True | Remove punctuation |
| `remove_numbers` | bool | False | Remove numeric characters |
| `remove_extra_whitespace` | bool | True | Normalize whitespace |
| `custom_stopwords` | list | None | Words to remove |

## Architecture

The `MultiHeadAttentionClassifier` implements a simplified version of the UIAN (User Intent Attention Network) architecture:

```
Input Text
    │
    ▼
┌─────────────────┐
│  Transformer    │
│  Encoder        │
│  (DistilBERT)   │
└────────┬────────┘
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
┌─────┐┌─────┐┌─────┐
│Head1││Head2││Head3│  ← Specialized Attention Heads
└──┬──┘└──┬──┘└──┬──┘
   │      │      │
   └──────┼──────┘
          │
          ▼
   ┌─────────────┐
   │   Fusion    │
   │   Layer     │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Classifier  │
   └─────────────┘
```

Each attention head learns to focus on different aspects of the input text, enabling:
- **Explainability**: Understand what the model focuses on
- **Modularity**: Different heads for different signal types
- **Robustness**: Multiple views of the same input

## Running Tests

```bash
# Run all tests
python -m pytest

# Run only traditional classifier tests
python -m pytest tests/test_classifier.py

# Run only neural classifier tests (requires torch, transformers)
python -m pytest tests/test_neural.py
```

## License

MIT
