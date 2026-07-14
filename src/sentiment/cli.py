
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
    train_p.add_argument("--sample-size", type=int, default=None, help="Randomly subsample this many rows before training (recommended for large datasets)")
    train_p.add_argument("--save-charts", action="store_true", help="Save confusion matrix images to charts/ instead of just printing text tables")

    predict_p = sub.add_parser("predict", help="Predict sentiment for a single sentence")
    predict_p.add_argument("text", help="The sentence to classify")
    predict_p.add_argument(
        "--model", choices=["classical", "roberta"], default="classical",
        help="'classical' uses your trained model (run `train` first); 'roberta' uses a pretrained transformer"
    )

    eval_roberta_p = sub.add_parser(
        "evaluate-roberta",
        help="Evaluate the pretrained RoBERTa model on a subset of the same held-out test split used for the classical models"
    )
    eval_roberta_p.add_argument("--data", default=None, help="Path to a reviews file (.csv or .json-lines); defaults to bundled sample data")
    eval_roberta_p.add_argument("--sample-size", type=int, default=None, help="Subsample this many rows before the train/test split (should match what you used for `train`)")
    eval_roberta_p.add_argument("--eval-limit", type=int, default=300, help="Max number of test examples to run through RoBERTa (CPU inference is slow, so this is capped by default)")

    mismatch_p = sub.add_parser(
        "analyze-mismatches",
        help="Find reviews where the text's word choice disagrees with the star-rating-derived sentiment label"
    )
    mismatch_p.add_argument("--data", default=None, help="Path to a reviews file (.csv or .json-lines); defaults to bundled sample data")
    mismatch_p.add_argument("--sample-size", type=int, default=None, help="Randomly subsample this many rows before analysis")
    mismatch_p.add_argument("--threshold", type=int, default=2, help="Minimum lexicon score difference to count as a mismatch (default: 2)")

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    from sentiment.evaluate import (
        print_comparison_table,
        print_confusion_matrix,
        print_misclassified,
        save_confusion_matrix_image,
    )
    from sentiment.pipeline import run_training

    results = run_training(csv_path=args.data, sample_size=args.sample_size)
    print("\nModel comparison:\n")
    print_comparison_table(results)

    if args.save_charts:
        from pathlib import Path
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)

    for result in results:
        if result.model_name != "baseline":
            print_confusion_matrix(result)
            print_misclassified(result)
            if args.save_charts:
                path = save_confusion_matrix_image(result, str(charts_dir / f"confusion_{result.model_name}.png"))
                print(f"Saved chart: {path}")

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


def cmd_evaluate_roberta(args: argparse.Namespace) -> int:
    from sentiment.evaluate import print_confusion_matrix, print_misclassified
    from sentiment.roberta import evaluate_roberta_on_test

    print(f"Running RoBERTa on up to {args.eval_limit} held-out test examples (this can take a few minutes on CPU)...\n")
    result, avg_ms, n_examples = evaluate_roberta_on_test(
        csv_path=args.data, sample_size=args.sample_size, eval_limit=args.eval_limit
    )

    print(f"Evaluated on {n_examples} examples\n")
    print(f"{'Model':<22}{'Accuracy':<12}{'F1 (macro)':<12}{'Avg time/example':<18}")
    print("-" * 64)
    print(f"{'roberta':<22}{result.accuracy:<12.3f}{result.f1_macro:<12.3f}{avg_ms:.1f} ms")

    print_confusion_matrix(result)
    print_misclassified(result)

    print(
        "\nCompare this against your classical model numbers from `python main.py train` "
        "-- same test split, same metrics, directly comparable."
    )
    return 0


def cmd_analyze_mismatches(args: argparse.Namespace) -> int:
    from sentiment.analysis import find_rating_text_mismatches, print_mismatch_report
    from sentiment.data import load_reviews

    df = load_reviews(args.data, sample_size=args.sample_size)
    report = find_rating_text_mismatches(df, min_word_diff=args.threshold)
    print_mismatch_report(report)
    return 0


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "evaluate-roberta":
        return cmd_evaluate_roberta(args)
    if args.command == "analyze-mismatches":
        return cmd_analyze_mismatches(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())