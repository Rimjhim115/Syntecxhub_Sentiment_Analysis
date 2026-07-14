from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentiment",
        description="Sentiment Analysis Tool — classical ML sentiment classification with an optional transformer comparison.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train and compare all classical models, save the best one")
    train_p.add_argument("--data", default=None, help="Path to a reviews file (.csv or .json-lines); defaults to bundled sample data")
    train_p.add_argument("--sample-size", type=int, default=None, help="Randomly subsample this many rows before training (recommended for large datasets)")  # ADD THIS LINE

    predict_p = sub.add_parser("predict", help="Predict sentiment for a single sentence")
    predict_p.add_argument("text", help="The sentence to classify")
    predict_p.add_argument(
        "--model", choices=["classical", "roberta"], default="classical",
        help="'classical' uses your trained model (run `train` first); 'roberta' uses a pretrained transformer"
    )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    from sentiment.evaluate import print_comparison_table, print_confusion_matrix, print_misclassified
    from sentiment.pipeline import run_training

    results = run_training(csv_path=args.data, sample_size=args.sample_size)  # CHANGED THIS LINE
    print("\nModel comparison:\n")
    print_comparison_table(results)

    for result in results:
        if result.model_name != "baseline":
            print_confusion_matrix(result)
            print_misclassified(result)

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if args.model == "roberta":
        from sentiment.roberta import predict_with_roberta
        result = predict_with_roberta(args.text)
        print(f"Sentiment: {result['sentiment']}  (confidence: {result['confidence']:.3f}, "
              f"{result['time_seconds']*1000:.1f} ms)")
    else:
        from sentiment.pipeline import predict_text
        try:
            result = predict_text(args.text)
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        conf = f" (confidence: {result['confidence']:.3f})" if result["confidence"] is not None else ""
        print(f"Sentiment: {result['sentiment']}{conf}")

    return 0


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())