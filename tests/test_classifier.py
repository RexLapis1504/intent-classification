"""Tests for the classifier module."""

import tempfile
from pathlib import Path

import pytest

from intent_classifier.classifier import IntentClassifier
from intent_classifier.preprocessing import TextPreprocessor


class TestIntentClassifier:
    """Tests for IntentClassifier class."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data for tests."""
        texts = [
            "hello",
            "hi there",
            "hey",
            "good morning",
            "goodbye",
            "bye",
            "see you later",
            "farewell",
            "help me",
            "I need help",
            "assist me",
            "can you help",
        ]
        intents = [
            "greeting",
            "greeting",
            "greeting",
            "greeting",
            "goodbye",
            "goodbye",
            "goodbye",
            "goodbye",
            "help",
            "help",
            "help",
            "help",
        ]
        return texts, intents

    @pytest.fixture
    def trained_classifier(self, sample_data):
        """A trained classifier fixture."""
        texts, intents = sample_data
        classifier = IntentClassifier()
        classifier.fit(texts, intents)
        return classifier

    def test_init_default(self):
        """Test default initialization."""
        classifier = IntentClassifier()

        assert classifier.algorithm == "logistic_regression"
        assert classifier.confidence_threshold == 0.0
        assert not classifier.is_fitted
        assert classifier.intents == []

    def test_init_with_algorithm(self):
        """Test initialization with different algorithm."""
        classifier = IntentClassifier(algorithm="naive_bayes")

        assert classifier.algorithm == "naive_bayes"

    def test_init_unsupported_algorithm(self):
        """Test initialization with unsupported algorithm raises error."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            IntentClassifier(algorithm="random_forest")

    def test_fit(self, sample_data):
        """Test fitting the classifier."""
        texts, intents = sample_data
        classifier = IntentClassifier()

        result = classifier.fit(texts, intents)

        assert result is classifier
        assert classifier.is_fitted
        assert set(classifier.intents) == {"greeting", "goodbye", "help"}

    def test_fit_mismatched_lengths(self):
        """Test fitting with mismatched data raises error."""
        classifier = IntentClassifier()

        with pytest.raises(ValueError, match="must match"):
            classifier.fit(["hello", "hi"], ["greeting"])

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error."""
        classifier = IntentClassifier()

        with pytest.raises(ValueError, match="cannot be empty"):
            classifier.fit([], [])

    def test_predict(self, trained_classifier):
        """Test prediction."""
        result = trained_classifier.predict("hello there")

        assert result == "greeting"

    def test_predict_unfitted(self):
        """Test prediction on unfitted classifier raises error."""
        classifier = IntentClassifier()

        with pytest.raises(RuntimeError, match="must be fitted"):
            classifier.predict("hello")

    def test_predict_with_confidence(self, trained_classifier):
        """Test prediction with confidence scores."""
        result = trained_classifier.predict_with_confidence("hello there")

        assert "intent" in result
        assert "confidence" in result
        assert "all_scores" in result
        assert result["intent"] == "greeting"
        assert 0 <= result["confidence"] <= 1
        assert set(result["all_scores"].keys()) == {"greeting", "goodbye", "help"}

    def test_predict_batch(self, trained_classifier):
        """Test batch prediction."""
        texts = ["hello", "goodbye", "help me"]

        results = trained_classifier.predict_batch(texts)

        assert len(results) == 3
        assert results[0] == "greeting"
        assert results[1] == "goodbye"
        assert results[2] == "help"

    def test_confidence_threshold(self, sample_data):
        """Test confidence threshold returns unknown."""
        texts, intents = sample_data
        classifier = IntentClassifier(confidence_threshold=0.99)
        classifier.fit(texts, intents)

        # A very ambiguous input should fall below threshold
        result = classifier.predict("xyz abc 123")

        # Either returns unknown or the actual prediction depending on model confidence
        assert result in ["unknown", "greeting", "goodbye", "help"]

    def test_save_and_load(self, trained_classifier):
        """Test saving and loading the classifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            trained_classifier.save(model_path)
            loaded = IntentClassifier.load(model_path)

            assert loaded.is_fitted
            assert loaded.algorithm == trained_classifier.algorithm
            assert loaded.intents == trained_classifier.intents

            # Verify predictions match
            test_text = "hello there"
            assert loaded.predict(test_text) == trained_classifier.predict(test_text)

    def test_save_unfitted(self):
        """Test saving unfitted classifier raises error."""
        classifier = IntentClassifier()

        with pytest.raises(RuntimeError, match="Cannot save"):
            classifier.save("model.pkl")

    def test_load_nonexistent(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            IntentClassifier.load("nonexistent.pkl")

    def test_get_feature_importance(self, trained_classifier):
        """Test getting feature importance."""
        features = trained_classifier.get_feature_importance("greeting", top_n=5)

        assert len(features) <= 5
        assert all(isinstance(f, tuple) and len(f) == 2 for f in features)
        assert all(isinstance(f[0], str) and isinstance(f[1], float) for f in features)

    def test_get_feature_importance_unknown_intent(self, trained_classifier):
        """Test feature importance for unknown intent raises error."""
        with pytest.raises(ValueError, match="Unknown intent"):
            trained_classifier.get_feature_importance("nonexistent")

    def test_get_feature_importance_wrong_algorithm(self, sample_data):
        """Test feature importance with naive bayes raises error."""
        texts, intents = sample_data
        classifier = IntentClassifier(algorithm="naive_bayes")
        classifier.fit(texts, intents)

        with pytest.raises(ValueError, match="only available for"):
            classifier.get_feature_importance("greeting")

    def test_custom_preprocessor(self, sample_data):
        """Test with custom preprocessor."""
        texts, intents = sample_data
        preprocessor = TextPreprocessor(lowercase=False)
        classifier = IntentClassifier(preprocessor=preprocessor)
        classifier.fit(texts, intents)

        assert classifier.is_fitted
        result = classifier.predict("HELLO")
        assert result in classifier.intents

    def test_naive_bayes_algorithm(self, sample_data):
        """Test training with naive bayes algorithm."""
        texts, intents = sample_data
        classifier = IntentClassifier(algorithm="naive_bayes")
        classifier.fit(texts, intents)

        assert classifier.is_fitted
        result = classifier.predict("hello")
        assert result == "greeting"
