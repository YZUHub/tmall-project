import pandas as pd
import numpy as np
from typing import Dict, List

def analyze_dataset(df: pd.DataFrame) -> Dict:
    """Analyze dataset and return statistics"""
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats['numerical_stats'] = df[numerical_cols].describe().to_dict()

    return stats

def generate_data_report(df: pd.DataFrame, output_path: str = None):
    """Generate and save data analysis report"""
    report = []

    # Basic information
    report.append("=== Dataset Overview ===")
    report.append(f"Total Records: {len(df)}")
    report.append(f"Total Features: {len(df.columns)}")

    # Missing values
    report.append("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col, count in missing.items():
        if count > 0:
            report.append(f"{col}: {count} ({missing_pct[col]}%)")

    # Save report
    report_text = "\n".join(report)
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)

    return report_text
