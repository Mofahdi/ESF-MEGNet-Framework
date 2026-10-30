import argparse
import csv
import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42

METRICS = ("test_MAE", "R2")


def extract_metrics(run_dir):
    run_dir = os.path.abspath(run_dir)
    rows = []
    for result_path in sorted(glob.glob(os.path.join(run_dir, "*", "fold_*", "results.json"))):
        fold_dir = os.path.dirname(result_path)
        model = os.path.basename(os.path.dirname(fold_dir))
        fold_name = os.path.basename(fold_dir)
        try:
            fold = int(fold_name.removeprefix("fold_"))
        except ValueError as error:
            raise ValueError(f"Unexpected fold directory name: {fold_dir}") from error

        with open(result_path, encoding="utf-8") as result_file:
            result = json.load(result_file)
        rows.append({
            "model": model,
            "fold": fold,
            **{metric: result[metric] for metric in METRICS},
        })

    if not rows:
        raise FileNotFoundError(f"No model/fold results.json files found under {run_dir}")

    return sorted(rows, key=lambda row: (row["model"], row["fold"]))


def save_test_mae_boxplot(metrics, output_path, ylabel="Test MAE"):
    models = sorted({row["model"] for row in metrics})
    values_by_model = [
        [row["test_MAE"] for row in metrics if row["model"] == model]
        for model in models
    ]

    figure, axis = plt.subplots(figsize=(3.5, 2.6))
    axis.boxplot(values_by_model, tick_labels=models)
    axis.tick_params(axis="both", labelsize=9)
    #axis.tick_params(axis="x", labelrotation=45)
    #axis.set_title("Test MAE by model across folds")
    #axis.set_xlabel("Model", fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Experiment directory containing <model>/fold_<n>/results.json",
    )
    parser.add_argument(
        "--output",
        help="CSV output path (default: <run_dir>/metrics.csv)",
    )
    parser.add_argument(
        "--plot-output",
        help="Box plot output path (default: <run_dir>/test_MAE_boxplot.pdf)",
    )
    parser.add_argument(
        "--ylabel",
        default="Test MAE",
        help="Y-axis label for the box plot (default: %(default)s)",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    metrics = extract_metrics(run_dir)
    output_path = args.output or os.path.join(run_dir, "metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    models = sorted({row["model"] for row in metrics})
    folds = sorted({row["fold"] for row in metrics})
    fold_columns = [f"fold_{fold}" for fold in folds]
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        for index, metric in enumerate(METRICS):
            if index:
                output_file.write("\n")
            output_file.write(f"{metric}\n")
            writer.writerow(["model", *fold_columns, "mean", "std"])
            for model in models:
                values_by_fold = {
                    row["fold"]: row[metric]
                    for row in metrics
                    if row["model"] == model
                }
                fold_values = [values_by_fold.get(fold) for fold in folds]
                available_values = [value for value in fold_values if value is not None]
                mean_value = statistics.mean(available_values) if available_values else ""
                std_value = statistics.stdev(available_values) if len(available_values) > 1 else ""
                writer.writerow([
                    model,
                    *(value if value is not None else "" for value in fold_values),
                    mean_value,
                    std_value,
                ])
    print(f"Saved metrics for {len(metrics)} model-fold runs to {output_path}")
    plot_path = args.plot_output or os.path.join(run_dir, "test_MAE_boxplot.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    save_test_mae_boxplot(metrics, plot_path, ylabel=args.ylabel)
    print(f"Saved per-model test_MAE box plot to {plot_path}")


if __name__ == "__main__":
    main()
