"""Text preprocessing utilities for intent classification."""

import re
import string
from typing import List, Optional


class TextPreprocessor:
    """Preprocesses text for intent classification.

    This class handles text normalization, tokenization, and cleaning
    to prepare text for classification.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        custom_stopwords: Optional[List[str]] = None,
    ):
        """Initialize the text preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation characters.
            remove_numbers: Whether to remove numeric characters.
            remove_extra_whitespace: Whether to normalize whitespace.
            custom_stopwords: Optional list of stopwords to remove.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.custom_stopwords = set(custom_stopwords) if custom_stopwords else set()

    def preprocess(self, text: str) -> str:
        """Preprocess a single text string.

        Args:
            text: The input text to preprocess.

        Returns:
            The preprocessed text.
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = " ".join(text.split())

        # Remove custom stopwords
        if self.custom_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.custom_stopwords]
            text = " ".join(words)

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of text strings.

        Args:
            texts: List of input texts to preprocess.

        Returns:
            List of preprocessed texts.
        """
        return [self.preprocess(text) for text in texts]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: The input text to tokenize.

        Returns:
            List of tokens.
        """
        preprocessed = self.preprocess(text)
        return preprocessed.split()
