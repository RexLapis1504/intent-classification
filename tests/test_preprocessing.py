"""Tests for the preprocessing module."""

import pytest

from intent_classifier.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_default_preprocessing(self):
        """Test default preprocessing settings."""
        preprocessor = TextPreprocessor()
        text = "Hello, World!  How are you?"

        result = preprocessor.preprocess(text)

        assert result == "hello world how are you"

    def test_lowercase_disabled(self):
        """Test with lowercase disabled."""
        preprocessor = TextPreprocessor(lowercase=False)
        text = "Hello World"

        result = preprocessor.preprocess(text)

        assert result == "Hello World"

    def test_remove_punctuation_disabled(self):
        """Test with punctuation removal disabled."""
        preprocessor = TextPreprocessor(remove_punctuation=False)
        text = "Hello, World!"

        result = preprocessor.preprocess(text)

        assert result == "hello, world!"

    def test_remove_numbers_enabled(self):
        """Test with number removal enabled."""
        preprocessor = TextPreprocessor(remove_numbers=True)
        text = "Call me at 555-1234"

        result = preprocessor.preprocess(text)

        assert result == "call me at"

    def test_extra_whitespace_removal(self):
        """Test extra whitespace is removed."""
        preprocessor = TextPreprocessor()
        text = "  Hello   World  "

        result = preprocessor.preprocess(text)

        assert result == "hello world"

    def test_custom_stopwords(self):
        """Test custom stopword removal."""
        preprocessor = TextPreprocessor(custom_stopwords=["the", "a", "an"])
        text = "the quick brown fox"

        result = preprocessor.preprocess(text)

        assert result == "quick brown fox"

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        preprocessor = TextPreprocessor()
        texts = ["Hello!", "World!"]

        results = preprocessor.preprocess_batch(texts)

        assert results == ["hello", "world"]

    def test_tokenize(self):
        """Test tokenization."""
        preprocessor = TextPreprocessor()
        text = "Hello, World!"

        tokens = preprocessor.tokenize(text)

        assert tokens == ["hello", "world"]

    def test_empty_string(self):
        """Test preprocessing empty string."""
        preprocessor = TextPreprocessor()

        result = preprocessor.preprocess("")

        assert result == ""

    def test_non_string_input(self):
        """Test preprocessing non-string input."""
        preprocessor = TextPreprocessor()

        result = preprocessor.preprocess(123)

        assert result == "123"

    def test_unicode_text(self):
        """Test preprocessing unicode text."""
        preprocessor = TextPreprocessor()
        text = "Caf\u00e9 au lait"

        result = preprocessor.preprocess(text)

        assert result == "caf\u00e9 au lait"
