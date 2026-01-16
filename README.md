# Intent Classification

A simple and extensible Python library for classifying text into predefined intents using machine learning.

## Features

- **Multiple Algorithms**: Support for Logistic Regression and Naive Bayes classifiers
- **Text Preprocessing**: Built-in text preprocessing with customizable options
- **Confidence Scores**: Get confidence scores for predictions
- **Model Persistence**: Save and load trained models
- **CLI Interface**: Command-line tool for training and prediction
- **Feature Importance**: Inspect which features contribute to classifications

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

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

# Load a saved model
loaded = IntentClassifier.load("model.pkl")
```

### Command Line Interface

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

### TextPreprocessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowercase` | bool | True | Convert text to lowercase |
| `remove_punctuation` | bool | True | Remove punctuation |
| `remove_numbers` | bool | False | Remove numeric characters |
| `remove_extra_whitespace` | bool | True | Normalize whitespace |
| `custom_stopwords` | list | None | Words to remove |

## Running Tests

```bash
pytest
```

## License

MIT
