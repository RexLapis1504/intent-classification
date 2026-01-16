"""Command-line interface for intent classification."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from .classifier import IntentClassifier


def load_training_data(path: Path) -> Tuple[List[str], List[str]]:
    """Load training data from a JSON file.

    Expected format:
    {
        "intents": [
            {
                "name": "intent_name",
                "examples": ["example1", "example2", ...]
            },
            ...
        ]
    }

    Args:
        path: Path to the JSON file.

    Returns:
        Tuple of (texts, intents) lists.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    intents = []
    for intent_data in data["intents"]:
        intent_name = intent_data["name"]
        for example in intent_data["examples"]:
            texts.append(example)
            intents.append(intent_name)

    return texts, intents


def train_command(args: argparse.Namespace) -> int:
    """Handle the train subcommand."""
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Training data file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"Loading training data from {data_path}...")
    texts, intents = load_training_data(data_path)
    print(f"Loaded {len(texts)} examples across {len(set(intents))} intents")

    print(f"Training classifier using {args.algorithm}...")
    classifier = IntentClassifier(
        algorithm=args.algorithm,
        confidence_threshold=args.threshold,
    )
    classifier.fit(texts, intents)

    output_path = Path(args.output)
    classifier.save(output_path)
    print(f"Model saved to {output_path}")

    return 0


def predict_command(args: argparse.Namespace) -> int:
    """Handle the predict subcommand."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    classifier = IntentClassifier.load(model_path)

    if args.text:
        # Single text prediction
        result = classifier.predict_with_confidence(args.text)
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")

        if args.verbose:
            print("\nAll scores:")
            sorted_scores = sorted(
                result["all_scores"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for intent, score in sorted_scores:
                print(f"  {intent}: {score:.4f}")
    else:
        # Interactive mode
        print("Intent Classifier - Interactive Mode")
        print("Enter text to classify (Ctrl+C to exit)")
        print("-" * 40)

        try:
            while True:
                text = input("\n> ").strip()
                if not text:
                    continue

                result = classifier.predict_with_confidence(text)
                print(f"Intent: {result['intent']} (confidence: {result['confidence']:.4f})")

                if args.verbose:
                    sorted_scores = sorted(
                        result["all_scores"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    print("Top predictions:", end=" ")
                    print(", ".join(f"{i}: {s:.2f}" for i, s in sorted_scores))

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")

    return 0


def info_command(args: argparse.Namespace) -> int:
    """Handle the info subcommand."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    classifier = IntentClassifier.load(model_path)

    print(f"Model: {model_path}")
    print(f"Algorithm: {classifier.algorithm}")
    print(f"Confidence Threshold: {classifier.confidence_threshold}")
    print(f"Number of Intents: {len(classifier.intents)}")
    print(f"\nIntents:")
    for intent in sorted(classifier.intents):
        print(f"  - {intent}")

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Intent Classification CLI",
        prog="intent-classify",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new classifier")
    train_parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to training data JSON file",
    )
    train_parser.add_argument(
        "-o", "--output",
        default="model.pkl",
        help="Output path for the trained model (default: model.pkl)",
    )
    train_parser.add_argument(
        "-a", "--algorithm",
        choices=["logistic_regression", "naive_bayes"],
        default="logistic_regression",
        help="Classification algorithm (default: logistic_regression)",
    )
    train_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.0,
        help="Confidence threshold (default: 0.0)",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict intent for text")
    predict_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to trained model file",
    )
    predict_parser.add_argument(
        "text",
        nargs="?",
        help="Text to classify (omit for interactive mode)",
    )
    predict_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all confidence scores",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to trained model file",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        return train_command(args)
    elif args.command == "predict":
        return predict_command(args)
    elif args.command == "info":
        return info_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
