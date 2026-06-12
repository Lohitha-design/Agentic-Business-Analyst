import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

import pandas as pd
import logging

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    threshold = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    logging.info("Dropped columns with >50% missing values")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    logging.info("Filled missing values with mean/mode")

    # Convert categorical columns into numeric (one-hot encoding)
    df = pd.get_dummies(df, drop_first=True)
    logging.info("Converted categorical columns to dummy variables")

    return df
