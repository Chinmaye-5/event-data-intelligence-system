import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os  # added

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(" Data loaded successfully")
        return data
    except Exception as e:
        print(" Error loading data:", e)
        return None

def preprocess_data(data):
    print("Preprocessing data...")

    # Drop missing values
    data = data.dropna()

    # Select only numeric columns for clustering
    numeric_data = data.select_dtypes(include=['int64', 'float64'])

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    print(" Preprocessing complete")
    return scaled_data, data

def apply_clustering(scaled_data, original_data, k=3):
    print("Applying K-means clustering...")

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    original_data['Cluster'] = clusters

    print("Clustering complete")
    return original_data

def main():
    file_path = "data/events.csv"  # make sure this file exists

    data = load_data(file_path)

    if data is None:
        print("Using sample dataset...")
        data = pd.DataFrame({
            "attendees": [100, 150, 200, 250, 300, 120, 180],
            "duration": [2, 3, 4, 5, 6, 2, 3],
            "rating": [4.5, 4.0, 4.8, 4.2, 4.9, 3.8, 4.3]
        })

    scaled_data, original_data = preprocess_data(data)

    clustered_data = apply_clustering(scaled_data, original_data)

    print("\n Clustered Data Sample:")
    print(clustered_data.head())

    os.makedirs("data", exist_ok=True)

    clustered_data.to_csv("data/output.csv", index=False)
    print(" Output saved to data/output.csv")

if __name__ == "__main__":
    main()
