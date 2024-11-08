import pandas as pd

# Define metrics and thresholds
metrics = {
    "Accuracy": {"threshold": 0.90, "value": 0.95},
    "F1 Score": {"threshold": 0.85, "value": 0.88},
    "Training Time (min)": {"threshold": 5, "value": 4.5}
}

# Prepare data for the table
table_data = []
for metric, info in metrics.items():
    threshold = info["threshold"]
    value = info["value"]
    status = "❌" if value <= threshold else "✔️"
    table_data.append([metric, threshold, value, status])

# Create a DataFrame and convert it to Markdown table format
df = pd.DataFrame(table_data, columns=["Metric", "Threshold", "Value", "Status"])

# Ensure columns are consistently left-aligned in the Markdown output
table_md = df.to_markdown(index=False, tablefmt="pipe", floatfmt=".2f")

# Print the formatted Markdown table for GitHub Action to capture
print("## Model Performance Metrics")
print(table_md)
