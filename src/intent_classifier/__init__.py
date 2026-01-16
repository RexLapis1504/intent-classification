"""Intent Classification Library.

A simple and extensible library for classifying text into predefined intents.

This library provides two classification approaches:
1. IntentClassifier: Traditional ML using TF-IDF + sklearn classifiers
2. MultiHeadAttentionClassifier: Neural approach with multi-head attention
   inspired by the UIAN (User Intent Attention Network) architecture
"""

from .classifier import IntentClassifier
from .preprocessing import TextPreprocessor

# Neural classifier is optional (requires torch and transformers)
try:
    from .neural import (
        MultiHeadAttentionClassifier,
        MultiHeadIntentNetwork,
        AttentionHead,
        DEFAULT_ATTENTION_HEADS,
    )
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False
    MultiHeadAttentionClassifier = None
    MultiHeadIntentNetwork = None
    AttentionHead = None
    DEFAULT_ATTENTION_HEADS = None

__version__ = "0.2.0"
__all__ = [
    "IntentClassifier",
    "TextPreprocessor",
    "MultiHeadAttentionClassifier",
    "MultiHeadIntentNetwork",
    "AttentionHead",
    "DEFAULT_ATTENTION_HEADS",
]


def is_neural_available() -> bool:
    """Check if neural classifier dependencies are available."""
    return _NEURAL_AVAILABLE
