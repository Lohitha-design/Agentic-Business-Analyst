import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

import pandas as pd
import logging

def validate_data(df: pd.DataFrame) -> dict:
    report = {
        "missing_values": int(df.isnull().sum().sum()),
        "invalid_types": [],
        "negative_values": False,
        "status": "valid"
    }

    # Missing values check
    if report["missing_values"] > 0:
        logging.warning(f"Data contains {report['missing_values']} missing values.")
        report["status"] = "invalid"

    # Numeric columns only
    numeric_df = df.select_dtypes(include="number")

    # Negative values check
    if (numeric_df < 0).any().any():
        logging.warning("Data contains negative values in numeric columns.")
        report["negative_values"] = True
        report["status"] = "invalid"

    # Type validation (non-numeric columns still existing)
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    if len(non_numeric_cols) > 0:
        logging.warning(f"Non-numeric columns found: {list(non_numeric_cols)}")
        report["invalid_types"] = list(non_numeric_cols)
        report["status"] = "invalid"

    if report["status"] == "valid":
        logging.info("Data validation passed successfully.")

    return report
