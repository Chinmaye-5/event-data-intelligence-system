from src.utils.data_loader import load_data
from src.services.clustering_service import preprocess_data, apply_clustering
from src.config.config import CONFIG
from src.utils.logger import log
import os

def run_pipeline():
    log("Starting pipeline...")

    data = load_data(CONFIG["DATA_PATH"])

    if data is None:
        log("Using fallback dataset...")
        import pandas as pd
        data = pd.DataFrame({
            "attendees": [100, 150, 200, 250, 300],
            "duration": [2, 3, 4, 5, 6],
            "rating": [4.5, 4.0, 4.8, 4.2, 4.9]
        })

    scaled, original = preprocess_data(data)

    result = apply_clustering(scaled, original, CONFIG["CLUSTERS"])

    os.makedirs("data", exist_ok=True)
    result.to_csv(CONFIG["OUTPUT_PATH"], index=False)

    log("Pipeline execution complete")
