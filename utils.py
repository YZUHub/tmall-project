import pandas as pd


def new_function():
    pass


def check_metrics(metrics):
    # Define metrics and thresholds
    metrics = {
        "Accuracy": {"threshold": 0.6687, "value": metrics["accuracy"]},
        "F1 Score": {"threshold": 0.6632, "value": metrics["f1"]},
        "AUC": {"threshold": 0.7134, "value": metrics["auc"]},
    }

    # Prepare data for the table
    table_data = []
    for metric, info in metrics.items():
        threshold = info["threshold"]
        value = info["value"]
        status = "❌" if value <= threshold else "✅"
        table_data.append([metric, threshold, value, status])

    # Create a DataFrame and convert it to Markdown table format
    df = pd.DataFrame(table_data, columns=["Metric", "Threshold", "Value", "Status"])

    # Ensure columns are consistently left-aligned in the Markdown output
    table_md = df.to_markdown(index=False, tablefmt="pipe", floatfmt=".2f")
    return table_md
