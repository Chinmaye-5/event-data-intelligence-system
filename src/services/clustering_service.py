from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.logger import log

def preprocess_data(data):
    log("Preprocessing data...")

    data = data.dropna()
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)

    log("Preprocessing complete")
    return scaled, data


def apply_clustering(scaled_data, original_data, k):
    log("Applying KMeans clustering...")

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(scaled_data)

    original_data["Cluster"] = clusters

    log("Clustering complete")
    return original_data
