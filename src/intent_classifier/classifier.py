"""Core intent classification module."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .preprocessing import TextPreprocessor


class IntentClassifier:
    """A machine learning-based intent classifier.

    This classifier uses TF-IDF vectorization combined with a classification
    algorithm to predict intents from text input.
    """

    SUPPORTED_ALGORITHMS = {
        "logistic_regression": LogisticRegression,
        "naive_bayes": MultinomialNB,
    }

    def __init__(
        self,
        algorithm: str = "logistic_regression",
        preprocessor: Optional[TextPreprocessor] = None,
        confidence_threshold: float = 0.0,
        **classifier_kwargs: Any,
    ):
        """Initialize the intent classifier.

        Args:
            algorithm: Classification algorithm to use. Supported options are
                'logistic_regression' and 'naive_bayes'.
            preprocessor: Optional TextPreprocessor instance. If None, a default
                preprocessor will be created.
            confidence_threshold: Minimum confidence score for predictions.
                Predictions below this threshold return 'unknown'.
            **classifier_kwargs: Additional keyword arguments passed to the
                underlying classifier.
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )

        self.algorithm = algorithm
        self.preprocessor = preprocessor or TextPreprocessor()
        self.confidence_threshold = confidence_threshold
        self.classifier_kwargs = classifier_kwargs

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._classifier: Optional[Any] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._is_fitted = False
        self._intents: List[str] = []

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
        vectorizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "IntentClassifier":
        """Train the classifier on the provided data.

        Args:
            texts: List of training text samples.
            intents: List of intent labels corresponding to each text.
            vectorizer_kwargs: Optional keyword arguments for TfidfVectorizer.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If texts and intents have different lengths.
        """
        if len(texts) != len(intents):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of intents ({len(intents)})"
            )

        if len(texts) == 0:
            raise ValueError("Training data cannot be empty")

        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)

        # Initialize and fit vectorizer
        vectorizer_kwargs = vectorizer_kwargs or {}
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            **vectorizer_kwargs,
        )
        X = self._vectorizer.fit_transform(processed_texts)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(intents)
        self._intents = list(self._label_encoder.classes_)

        # Initialize and fit classifier
        classifier_class = self.SUPPORTED_ALGORITHMS[self.algorithm]
        default_kwargs = self._get_default_classifier_kwargs()
        merged_kwargs = {**default_kwargs, **self.classifier_kwargs}
        self._classifier = classifier_class(**merged_kwargs)
        self._classifier.fit(X, y)

        self._is_fitted = True
        return self

    def _get_default_classifier_kwargs(self) -> Dict[str, Any]:
        """Get default kwargs for the selected classifier."""
        if self.algorithm == "logistic_regression":
            return {"max_iter": 1000, "random_state": 42}
        return {}

    def predict(self, text: str) -> str:
        """Predict the intent for a single text input.

        Args:
            text: The input text to classify.

        Returns:
            The predicted intent label, or 'unknown' if confidence
            is below the threshold.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        result = self.predict_with_confidence(text)
        return result["intent"]

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """Predict intent with confidence scores for a single text.

        Args:
            text: The input text to classify.

        Returns:
            Dictionary containing:
                - intent: The predicted intent (or 'unknown')
                - confidence: Confidence score for the prediction
                - all_scores: Dict mapping all intents to their scores

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")

        # Preprocess and vectorize
        processed_text = self.preprocessor.preprocess(text)
        X = self._vectorizer.transform([processed_text])

        # Get prediction probabilities
        proba = self._classifier.predict_proba(X)[0]
        predicted_idx = np.argmax(proba)
        confidence = proba[predicted_idx]

        # Map scores to intent names
        all_scores = {
            intent: float(score)
            for intent, score in zip(self._intents, proba)
        }

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            intent = "unknown"
        else:
            intent = self._intents[predicted_idx]

        return {
            "intent": intent,
            "confidence": float(confidence),
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predict intents for multiple texts.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of predicted intent labels.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        return [self.predict(text) for text in texts]

    def predict_batch_with_confidence(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Predict intents with confidence scores for multiple texts.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of prediction dictionaries.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        return [self.predict_with_confidence(text) for text in texts]

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained classifier to disk.

        Args:
            path: File path to save the model (should end in .pkl).

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "algorithm": self.algorithm,
            "confidence_threshold": self.confidence_threshold,
            "classifier_kwargs": self.classifier_kwargs,
            "preprocessor": self.preprocessor,
            "vectorizer": self._vectorizer,
            "classifier": self._classifier,
            "label_encoder": self._label_encoder,
            "intents": self._intents,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IntentClassifier":
        """Load a trained classifier from disk.

        Args:
            path: File path to load the model from.

        Returns:
            A fitted IntentClassifier instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            algorithm=model_data["algorithm"],
            preprocessor=model_data["preprocessor"],
            confidence_threshold=model_data["confidence_threshold"],
            **model_data["classifier_kwargs"],
        )
        instance._vectorizer = model_data["vectorizer"]
        instance._classifier = model_data["classifier"]
        instance._label_encoder = model_data["label_encoder"]
        instance._intents = model_data["intents"]
        instance._is_fitted = True

        return instance

    def get_feature_importance(
        self, intent: str, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Get the most important features for a specific intent.

        Only available for logistic regression classifier.

        Args:
            intent: The intent to get feature importance for.
            top_n: Number of top features to return.

        Returns:
            List of (feature, importance) tuples sorted by importance.

        Raises:
            RuntimeError: If classifier is not fitted.
            ValueError: If intent is not known or algorithm doesn't support this.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted first")

        if self.algorithm != "logistic_regression":
            raise ValueError(
                "Feature importance is only available for logistic_regression"
            )

        if intent not in self._intents:
            raise ValueError(f"Unknown intent: {intent}")

        intent_idx = self._intents.index(intent)
        feature_names = self._vectorizer.get_feature_names_out()

        # Handle binary vs multiclass
        if len(self._intents) == 2:
            coefficients = self._classifier.coef_[0]
            if intent_idx == 0:
                coefficients = -coefficients
        else:
            coefficients = self._classifier.coef_[intent_idx]

        # Get top features
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        return [
            (feature_names[i], float(coefficients[i]))
            for i in top_indices
        ]
