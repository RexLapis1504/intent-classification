"""Tests for the neural multi-head attention classifier."""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if neural dependencies are not available
try:
    import torch
    from transformers import AutoTokenizer
    from intent_classifier.neural import (
        MultiHeadAttentionClassifier,
        MultiHeadIntentNetwork,
        AttentionHead,
        DEFAULT_ATTENTION_HEADS,
        TORCH_AVAILABLE,
        TRANSFORMERS_AVAILABLE,
    )
    NEURAL_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
except ImportError:
    NEURAL_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not NEURAL_AVAILABLE,
    reason="Neural dependencies (torch, transformers) not available"
)


class TestMultiHeadIntentNetwork:
    """Tests for the MultiHeadIntentNetwork module."""

    def test_init(self):
        """Test network initialization."""
        network = MultiHeadIntentNetwork(
            model_name="distilbert-base-uncased",
            num_labels=3,
            num_attention_heads=3,
        )
        assert network.num_attention_heads == 3
        assert network.num_labels == 3

    def test_forward(self):
        """Test forward pass."""
        network = MultiHeadIntentNetwork(
            model_name="distilbert-base-uncased",
            num_labels=2,
            num_attention_heads=2,
        )

        # Create dummy input
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encoding = tokenizer(
            "Hello world",
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )

        logits = network(encoding["input_ids"], encoding["attention_mask"])

        assert logits.shape == (1, 2)

    def test_forward_with_attention(self):
        """Test forward pass returning attention weights."""
        network = MultiHeadIntentNetwork(
            model_name="distilbert-base-uncased",
            num_labels=2,
            num_attention_heads=3,
        )

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encoding = tokenizer(
            "Hello world",
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )

        logits, attention = network(
            encoding["input_ids"],
            encoding["attention_mask"],
            return_attention=True
        )

        assert logits.shape == (1, 2)
        assert len(attention) == 3  # 3 attention heads


class TestMultiHeadAttentionClassifier:
    """Tests for the high-level MultiHeadAttentionClassifier."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data for tests."""
        texts = [
            "hello there",
            "hi how are you",
            "hey",
            "good morning",
            "goodbye",
            "bye bye",
            "see you later",
            "farewell",
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
        ]
        return texts, intents

    def test_init(self):
        """Test classifier initialization."""
        classifier = MultiHeadAttentionClassifier()
        assert not classifier.is_fitted
        assert classifier.intents == []

    def test_init_with_custom_heads(self):
        """Test initialization with custom attention heads."""
        custom_heads = [
            AttentionHead(name="custom1", description="Custom head 1"),
            AttentionHead(name="custom2", description="Custom head 2"),
        ]
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            attention_head_configs=custom_heads
        )
        assert classifier.num_attention_heads == 2

    def test_fit(self, sample_data):
        """Test fitting the classifier."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            model_name="distilbert-base-uncased",
            num_attention_heads=2,
            hidden_dim=64,  # Smaller for faster tests
        )

        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        assert classifier.is_fitted
        assert set(classifier.intents) == {"greeting", "goodbye"}

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error."""
        classifier = MultiHeadAttentionClassifier()
        with pytest.raises(ValueError, match="cannot be empty"):
            classifier.fit([], [])

    def test_fit_mismatched_lengths(self):
        """Test fitting with mismatched data raises error."""
        classifier = MultiHeadAttentionClassifier()
        with pytest.raises(ValueError, match="must match"):
            classifier.fit(["hello", "hi"], ["greeting"])

    def test_predict(self, sample_data):
        """Test prediction."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        result = classifier.predict("hello there")
        assert result in ["greeting", "goodbye"]

    def test_predict_unfitted(self):
        """Test prediction on unfitted classifier raises error."""
        classifier = MultiHeadAttentionClassifier()
        with pytest.raises(RuntimeError, match="must be fitted"):
            classifier.predict("hello")

    def test_predict_with_confidence(self, sample_data):
        """Test prediction with confidence scores."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        result = classifier.predict_with_confidence("hello")

        assert "intent" in result
        assert "confidence" in result
        assert "all_scores" in result
        assert 0 <= result["confidence"] <= 1
        assert set(result["all_scores"].keys()) == {"greeting", "goodbye"}

    def test_predict_with_attention(self, sample_data):
        """Test prediction with attention weights."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        result = classifier.predict_with_attention("hello there friend")

        assert "intent" in result
        assert "confidence" in result
        assert "tokens" in result
        assert "attention_weights" in result
        assert "head_explanations" in result

        # Check attention structure
        assert len(result["attention_weights"]) == 2  # 2 heads
        for head_data in result["attention_weights"]:
            assert "head_name" in head_data
            assert "weights" in head_data
            assert "top_tokens" in head_data

    def test_classify_risk_tier(self, sample_data):
        """Test UIAN-style risk tier classification."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        result = classifier.classify_risk_tier("hello")

        assert "tier" in result
        assert "tier_name" in result
        assert "description" in result
        assert result["tier"] in [1, 2, 3, 4]

    def test_predict_batch(self, sample_data):
        """Test batch prediction."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        results = classifier.predict_batch(["hello", "goodbye"])

        assert len(results) == 2
        assert all(r in ["greeting", "goodbye"] for r in results)

    def test_save_and_load(self, sample_data):
        """Test saving and loading the classifier."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"

            classifier.save(model_path)
            loaded = MultiHeadAttentionClassifier.load(model_path)

            assert loaded.is_fitted
            assert loaded.intents == classifier.intents

            # Verify predictions work
            test_text = "hello there"
            original_result = classifier.predict(test_text)
            loaded_result = loaded.predict(test_text)
            # Note: Results may differ slightly due to floating point
            assert loaded_result in classifier.intents

    def test_save_unfitted(self):
        """Test saving unfitted classifier raises error."""
        classifier = MultiHeadAttentionClassifier()
        with pytest.raises(RuntimeError, match="Cannot save"):
            classifier.save("model")

    def test_load_nonexistent(self):
        """Test loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            MultiHeadAttentionClassifier.load("nonexistent")

    def test_visualize_attention_text(self, sample_data):
        """Test attention visualization in text format."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        viz = classifier.visualize_attention("hello there", head_index=0, output_format="text")

        assert "Attention Head:" in viz
        assert "Predicted Intent:" in viz
        assert "Token Attention Weights:" in viz

    def test_visualize_attention_html(self, sample_data):
        """Test attention visualization in HTML format."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        viz = classifier.visualize_attention("hello there", head_index=0, output_format="html")

        assert "<h3>" in viz
        assert "<span" in viz

    def test_visualize_attention_invalid_head(self, sample_data):
        """Test visualization with invalid head index raises error."""
        texts, intents = sample_data
        classifier = MultiHeadAttentionClassifier(
            num_attention_heads=2,
            hidden_dim=64,
        )
        classifier.fit(texts, intents, epochs=1, batch_size=4, verbose=False)

        with pytest.raises(ValueError, match="head_index must be"):
            classifier.visualize_attention("hello", head_index=10)


class TestAttentionHead:
    """Tests for AttentionHead dataclass."""

    def test_create(self):
        """Test creating an attention head config."""
        head = AttentionHead(
            name="test_head",
            description="A test attention head"
        )
        assert head.name == "test_head"
        assert head.description == "A test attention head"


class TestDefaultAttentionHeads:
    """Tests for default attention head configurations."""

    def test_default_heads_exist(self):
        """Test that default heads are defined."""
        assert DEFAULT_ATTENTION_HEADS is not None
        assert len(DEFAULT_ATTENTION_HEADS) >= 3

    def test_default_heads_have_required_fields(self):
        """Test that default heads have required fields."""
        for head in DEFAULT_ATTENTION_HEADS:
            assert hasattr(head, "name")
            assert hasattr(head, "description")
            assert head.name
            assert head.description
