from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_assets(risk_features: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.Series:
    scaled = StandardScaler().fit_transform(risk_features)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(scaled)
    return pd.Series(labels, index=risk_features.index, name="cluster")
