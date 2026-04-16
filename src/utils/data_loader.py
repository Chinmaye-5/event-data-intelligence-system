import pandas as pd
from src.utils.logger import log

def load_data(path):
    try:
        data = pd.read_csv(path)
        log("Data loaded successfully")
        return data
    except Exception as e:
        log(f"Error loading data: {e}")
        return None
