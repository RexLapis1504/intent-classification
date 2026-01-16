"""Neural multi-head attention classifier inspired by UIAN architecture.

This module implements a simplified version of the User Intent Attention Network (UIAN)
that uses multi-head attention to analyze different aspects of text content for
intent classification with explainability.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Check for torch availability
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
    _NNModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object  # Placeholder for class inheritance
    _NNModuleBase = object  # Placeholder for nn.Module inheritance

# Check for transformers availability
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModel = None
    AutoTokenizer = None


def _check_dependencies():
    """Check if required dependencies are available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MultiHeadAttentionClassifier. "
            "Install with: pip install torch"
        )
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers library is required for MultiHeadAttentionClassifier. "
            "Install with: pip install transformers"
        )


@dataclass
class AttentionHead:
    """Configuration for a specialized attention head."""
    name: str
    description: str


# Default attention heads inspired by UIAN
DEFAULT_ATTENTION_HEADS = [
    AttentionHead(
        name="manipulation",
        description="Detects linguistic manipulation patterns (sensationalism, emotional triggers)"
    ),
    AttentionHead(
        name="credibility",
        description="Assesses source credibility signals in text"
    ),
    AttentionHead(
        name="intent",
        description="Identifies intent markers and purpose signals"
    ),
]


class IntentDataset(Dataset):
    """PyTorch Dataset for intent classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class MultiHeadIntentNetwork(_NNModuleBase):
    """Neural network with multiple specialized attention heads for intent classification.

    This architecture is inspired by the UIAN (User Intent Attention Network) concept,
    using multiple attention heads to capture different aspects of text that signal
    user intent.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        num_attention_heads: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        _check_dependencies()

        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels

        # Load pre-trained transformer encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Specialized attention heads
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.encoder_dim,
                num_heads=4,  # Internal heads within each specialized head
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_attention_heads)
        ])

        # Head-specific projection layers
        self.head_projections = nn.ModuleList([
            nn.Linear(self.encoder_dim, hidden_dim)
            for _ in range(num_attention_heads)
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_attention_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        return_attention: bool = False
    ) -> Union[Any, Tuple[Any, List[Any]]]:
        """Forward pass through the network.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            logits: Classification logits [batch_size, num_labels]
            attention_weights: (optional) List of attention weights per head
        """
        # Get encoder outputs
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        hidden_states = encoder_output.last_hidden_state  # [batch, seq_len, hidden]

        # Apply each specialized attention head
        head_outputs = []
        attention_weights = []

        # Create key padding mask (True for padding tokens)
        key_padding_mask = ~attention_mask.bool()

        for i, (attn_head, projection) in enumerate(
            zip(self.attention_heads, self.head_projections)
        ):
            # Self-attention with the specialized head
            attn_output, attn_weight = attn_head(
                hidden_states,
                hidden_states,
                hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True
            )

            # Pool using CLS token (first position)
            pooled = attn_output[:, 0, :]

            # Project to head-specific space
            projected = self.dropout(F.relu(projection(pooled)))

            head_outputs.append(projected)
            attention_weights.append(attn_weight)

        # Concatenate all head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)

        # Fuse
        fused = self.fusion(multi_head_output)

        # Classify
        logits = self.classifier(fused)

        if return_attention:
            return logits, attention_weights
        return logits


class MultiHeadAttentionClassifier:
    """High-level interface for multi-head attention intent classification.

    This classifier uses a transformer-based architecture with multiple
    specialized attention heads to classify text into intents while
    providing explainable attention patterns.

    Inspired by the UIAN (User Intent Attention Network) architecture,
    this implementation focuses on content-based signals that can be
    extracted from text alone.

    Example:
        >>> classifier = MultiHeadAttentionClassifier()
        >>> classifier.fit(texts, labels, epochs=3)
        >>> result = classifier.predict_with_attention("Hello there!")
        >>> print(result["intent"])
        >>> print(result["attention_weights"])
    """

    # Tier thresholds for risk classification (UIAN-style)
    TIER_THRESHOLDS = {
        "tier_1_orchestrator": 0.8,
        "tier_2_believer": 0.5,
        "tier_3_casual": 0.2,
    }

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_attention_heads: int = 3,
        attention_head_configs: Optional[List[AttentionHead]] = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        max_length: int = 128,
        confidence_threshold: float = 0.0,
        device: Optional[str] = None,
    ):
        """Initialize the multi-head attention classifier.

        Args:
            model_name: Pre-trained transformer model to use as encoder.
            num_attention_heads: Number of specialized attention heads.
            attention_head_configs: Optional configurations for each head.
            hidden_dim: Dimension of hidden layers.
            dropout: Dropout probability.
            max_length: Maximum sequence length for tokenization.
            confidence_threshold: Minimum confidence for predictions.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        _check_dependencies()

        self.model_name = model_name
        self.num_attention_heads = num_attention_heads
        self.attention_head_configs = attention_head_configs or DEFAULT_ATTENTION_HEADS[:num_attention_heads]
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model will be initialized during fit
        self._model: Optional[MultiHeadIntentNetwork] = None
        self._intents: List[str] = []
        self._is_fitted = False

    @property
    def intents(self) -> List[str]:
        """Return list of known intents."""
        return self._intents.copy()

    @property
    def is_fitted(self) -> bool:
        """Return whether the classifier has been trained."""
        return self._is_fitted

    def fit(
        self,
        texts: List[str],
        intents: List[str],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        validation_split: float = 0.1,
        verbose: bool = True,
    ) -> "MultiHeadAttentionClassifier":
        """Train the classifier on the provided data.

        Args:
            texts: List of training text samples.
            intents: List of intent labels corresponding to each text.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            validation_split: Fraction of data to use for validation.
            verbose: Whether to print training progress.

        Returns:
            Self, for method chaining.
        """
        if len(texts) != len(intents):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of intents ({len(intents)})"
            )

        if len(texts) == 0:
            raise ValueError("Training data cannot be empty")

        # Create label mapping
        unique_intents = sorted(set(intents))
        self._intents = unique_intents
        intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
        labels = [intent_to_idx[intent] for intent in intents]

        # Initialize model
        self._model = MultiHeadIntentNetwork(
            model_name=self.model_name,
            num_labels=len(unique_intents),
            num_attention_heads=self.num_attention_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        # Split data
        n_samples = len(texts)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        val_indices = indices[:n_val] if n_val > 0 else []
        train_indices = indices[n_val:]

        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]

        # Create dataset and dataloader
        train_dataset = IntentDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Validation data
        if n_val > 0:
            val_texts = [texts[i] for i in val_indices]
            val_labels = [labels[i] for i in val_indices]
            val_dataset = IntentDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self._model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self._model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}"

                # Validation
                if n_val > 0:
                    val_acc = self._evaluate(val_loader)
                    msg += f", Val Acc: {val_acc:.4f}"

                print(msg)

        self._is_fitted = True
        return self

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on a dataloader."""
        self._model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                logits = self._model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

        self._model.train()
        return correct / total

    def predict(self, text: str) -> str:
        """Predict the intent for a single text input.

        Args:
            text: The input text to classify.

        Returns:
            The predicted intent label.
        """
        result = self.predict_with_confidence(text)
        return result["intent"]

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """Predict intent with confidence scores.

        Args:
            text: The input text to classify.

        Returns:
            Dictionary with intent, confidence, and all scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")

        self._model.eval()

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        # Map scores to intent names
        all_scores = {
            intent: float(probs[i])
            for i, intent in enumerate(self._intents)
        }

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            intent = "unknown"
        else:
            intent = self._intents[predicted_idx]

        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": all_scores,
        }

    def predict_with_attention(self, text: str) -> Dict[str, Any]:
        """Predict intent with attention weights for explainability.

        This method returns attention weights from each specialized head,
        which can be used to understand what aspects of the text influenced
        the classification decision.

        Args:
            text: The input text to classify.

        Returns:
            Dictionary containing:
                - intent: Predicted intent
                - confidence: Confidence score
                - all_scores: All intent scores
                - attention_weights: Attention weights per head
                - tokens: Tokenized input for alignment
                - head_explanations: Human-readable head descriptions
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")

        self._model.eval()

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, attention_weights = self._model(
                input_ids, attention_mask, return_attention=True
            )
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        # Get actual tokens (excluding padding)
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().numpy()
        )
        seq_len = attention_mask[0].sum().item()
        tokens = tokens[:seq_len]

        # Process attention weights
        processed_attention = []
        for i, attn in enumerate(attention_weights):
            # attn shape: [batch, seq, seq] -> take first batch, CLS row
            attn_weights = attn[0, 0, :seq_len].cpu().numpy()
            processed_attention.append({
                "head_name": self.attention_head_configs[i].name if i < len(self.attention_head_configs) else f"head_{i}",
                "weights": attn_weights.tolist(),
                "top_tokens": self._get_top_attended_tokens(tokens, attn_weights, top_k=5)
            })

        # Map scores to intent names
        all_scores = {
            intent: float(probs[i])
            for i, intent in enumerate(self._intents)
        }

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            intent = "unknown"
        else:
            intent = self._intents[predicted_idx]

        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": all_scores,
            "tokens": tokens,
            "attention_weights": processed_attention,
            "head_explanations": [
                {"name": h.name, "description": h.description}
                for h in self.attention_head_configs[:self.num_attention_heads]
            ]
        }

    def _get_top_attended_tokens(
        self,
        tokens: List[str],
        weights: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get the top attended tokens."""
        indices = np.argsort(weights)[-top_k:][::-1]
        return [(tokens[i], float(weights[i])) for i in indices if i < len(tokens)]

    def classify_risk_tier(self, text: str) -> Dict[str, Any]:
        """Classify text into UIAN-style risk tiers.

        This method maps the confidence score to a 4-tier risk system:
        - Tier 1 (Orchestrator): score > 0.8
        - Tier 2 (Committed Believer): 0.5 < score <= 0.8
        - Tier 3 (Casual Sharer): 0.2 < score <= 0.5
        - Tier 4 (Low Risk): score <= 0.2

        Args:
            text: The input text to classify.

        Returns:
            Dictionary with tier, description, and score.
        """
        result = self.predict_with_confidence(text)
        score = result["confidence"]

        if score > self.TIER_THRESHOLDS["tier_1_orchestrator"]:
            tier = 1
            tier_name = "Orchestrator"
            description = "High confidence classification - clear intent signals"
        elif score > self.TIER_THRESHOLDS["tier_2_believer"]:
            tier = 2
            tier_name = "Committed"
            description = "Moderate-high confidence - consistent intent patterns"
        elif score > self.TIER_THRESHOLDS["tier_3_casual"]:
            tier = 3
            tier_name = "Casual"
            description = "Moderate confidence - some intent signals present"
        else:
            tier = 4
            tier_name = "Low Risk"
            description = "Low confidence - minimal intent signals"

        return {
            "tier": tier,
            "tier_name": tier_name,
            "description": description,
            "intent": result["intent"],
            "confidence": score,
            "all_scores": result["all_scores"]
        }

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predict intents for multiple texts."""
        return [self.predict(text) for text in texts]

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained classifier.

        Args:
            path: Directory path to save the model.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self._model.state_dict(), path / "model.pt")

        # Save configuration
        config = {
            "model_name": self.model_name,
            "num_attention_heads": self.num_attention_heads,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "confidence_threshold": self.confidence_threshold,
            "intents": self._intents,
            "attention_head_configs": [
                {"name": h.name, "description": h.description}
                for h in self.attention_head_configs
            ]
        }

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultiHeadAttentionClassifier":
        """Load a trained classifier.

        Args:
            path: Directory path to load the model from.

        Returns:
            A fitted MultiHeadAttentionClassifier instance.
        """
        _check_dependencies()

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        # Load configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Create instance
        attention_configs = [
            AttentionHead(name=h["name"], description=h["description"])
            for h in config["attention_head_configs"]
        ]

        instance = cls(
            model_name=config["model_name"],
            num_attention_heads=config["num_attention_heads"],
            attention_head_configs=attention_configs,
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            max_length=config["max_length"],
            confidence_threshold=config["confidence_threshold"],
        )

        instance._intents = config["intents"]

        # Initialize and load model
        instance._model = MultiHeadIntentNetwork(
            model_name=config["model_name"],
            num_labels=len(instance._intents),
            num_attention_heads=config["num_attention_heads"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        ).to(instance.device)

        instance._model.load_state_dict(
            torch.load(path / "model.pt", map_location=instance.device)
        )
        instance._is_fitted = True

        return instance

    def visualize_attention(
        self,
        text: str,
        head_index: int = 0,
        output_format: str = "text"
    ) -> str:
        """Visualize attention weights for a text.

        Args:
            text: Input text to analyze.
            head_index: Which attention head to visualize.
            output_format: 'text' for ASCII, 'html' for HTML output.

        Returns:
            Formatted attention visualization.
        """
        result = self.predict_with_attention(text)

        if head_index >= len(result["attention_weights"]):
            raise ValueError(f"head_index must be < {len(result['attention_weights'])}")

        head_data = result["attention_weights"][head_index]
        tokens = result["tokens"]
        weights = head_data["weights"]

        if output_format == "text":
            lines = [
                f"Attention Head: {head_data['head_name']}",
                f"Predicted Intent: {result['intent']} ({result['confidence']:.2%})",
                "-" * 50,
                "Token Attention Weights:",
            ]

            for token, weight in zip(tokens, weights):
                bar_length = int(weight * 40)
                bar = "#" * bar_length
                lines.append(f"  {token:15s} {weight:.4f} |{bar}")

            lines.append("-" * 50)
            lines.append("Top Attended Tokens:")
            for token, weight in head_data["top_tokens"]:
                lines.append(f"  {token}: {weight:.4f}")

            return "\n".join(lines)

        elif output_format == "html":
            html_parts = [
                f"<h3>Attention Head: {head_data['head_name']}</h3>",
                f"<p>Predicted: <strong>{result['intent']}</strong> ({result['confidence']:.2%})</p>",
                "<div style='font-family: monospace;'>"
            ]

            for token, weight in zip(tokens, weights):
                # Color intensity based on weight
                intensity = int(255 * (1 - weight))
                color = f"rgb(255, {intensity}, {intensity})"
                html_parts.append(
                    f"<span style='background-color: {color}; padding: 2px; margin: 1px;'>"
                    f"{token}</span>"
                )

            html_parts.append("</div>")
            return "\n".join(html_parts)

        else:
            raise ValueError(f"Unknown output_format: {output_format}")
