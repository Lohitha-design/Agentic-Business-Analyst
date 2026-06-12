import logging
from src.Analysis.unsupervised import run_analysis
from src.Preprocessing.clean_data import clean_dataset
from src.Preprocessing.validate_data import validate_data
from agent.loop import run_agent
from src.Analysis.correlation_regression import correlation_analysis, regression_analysis
logging.basicConfig(level=logging.INFO)

def main(df,target_column) -> str:
    report = validate_data(df)
    if report["status"] == "invalid":
        df = clean_dataset(df)

    features = correlation_analysis(df, target=target_column, method="pearson", top_n=2)

    insights = regression_analysis(df,target=target_column,positive_features=features["positive_features"],negative_features=features["negative_features"])

    unsupervised_data = run_analysis(df)

    logging.info("Starting agent pipeline...")
    final_strategy = run_agent(insights, features, unsupervised_data)
    logging.info("Pipeline completed.")

    return final_strategy


