import pandas as pd
import logging
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

# Correlation analysis to identify relationships with target variable

def correlation_analysis(df: pd.DataFrame, target: str, method: str = "pearson", top_n: int = 2):
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Target column '{target}' must be numeric for correlation analysis")

    corr_matrix = df.corr(method=method)

    target_corr = corr_matrix[target].drop(target)

    positive_features = target_corr[target_corr > 0].sort_values(ascending=False).head(top_n).to_dict()
    negative_features = target_corr[target_corr < 0].sort_values().head(top_n).to_dict()

    logging.info(f"Correlation analysis completed using {method} method")

    return {"positive_features": positive_features, "negative_features": negative_features}

# Regression analysis to quantify relationships with target variable

def regression_analysis(df: pd.DataFrame, target: str, positive_features: dict, negative_features: dict):
    

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(f"Target column '{target}' must be numeric for regression analysis")

    X = df.drop([target], axis=1)
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    coefficients = dict(zip(X.columns, model.coef_))

    important_features = list(set(positive_features.keys()) | set(negative_features.keys()))

    results = {
        "intercept": model.intercept_,
        "r2_score": model.score(X, y),
        "coefficients": coefficients,
        "important_features": {
            feature: coefficients[feature]
            for feature in important_features if feature in coefficients
        },
        "positive_features": positive_features,   
        "negative_features": negative_features   
    }

    return results
