import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest

def run_analysis(df, n_clusters=3, contamination=0.05):

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df["Cluster"] = clusters

    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, clusters)
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        "Feature": df.columns[:-1],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    
    iso = IsolationForest(random_state=42, contamination=contamination)
    anomaly_scores = iso.fit_predict(X)
    df["Anomaly"] = anomaly_scores  
    
    results = {
        "clusters_summary": df["Cluster"].value_counts().to_dict(),
        "feature_importance": feature_importance.to_dict(orient="records"),
        "anomalies": df[df["Anomaly"] == -1].to_dict(orient="records")
    }

    return results

