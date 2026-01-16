"""Intent Classification Library.

A simple and extensible library for classifying text into predefined intents.
"""

from .classifier import IntentClassifier
from .preprocessing import TextPreprocessor

__version__ = "0.1.0"
__all__ = ["IntentClassifier", "TextPreprocessor"]
