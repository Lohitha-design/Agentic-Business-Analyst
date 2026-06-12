import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_file(file) -> pd.DataFrame:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file,encoding_errors="ignore")
        elif file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file,encoding_errors="ignore")
        elif file.name.endswith(".json"):
            df = pd.read_json(file,encoding_errors="ignore")
        else:
            raise ValueError(f"Unsupported file format: {file.name}")

        logging.info(f"File loaded successfully: {file.name}")
        return df

    except Exception as e:
        logging.error(f"Error loading file {file.name}: {e}")
        raise


