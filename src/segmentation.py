"""Customer segmentation with K-Means.

Shared by the notebook and the website export:

    python -m src.segmentation        # train, save model, write docs/data/model.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
RANDOM_STATE = 42
K = 5

FEATURES = ["income", "spending"]
COLUMN_NAMES = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Age": "age",
    "Annual Income (k$)": "income",
    "Spending Score (1-100)": "spending",
}

# Business meaning of each income/spending quadrant. Clusters are matched to
# these by the position of their centre, so names stay correct even if
# K-Means numbers the clusters differently on another run.
PERSONAS = {
    ("high", "high"): {
        "name": "Premium Spenders",
        "tagline": "High income, high spending",
        "description": "The mall's most valuable customers. They earn well and spend freely.",
        "strategy": [
            "VIP loyalty tier with exclusive perks",
            "Early access to new collections and events",
            "Personal shopping and premium brand partnerships",
        ],
    },
    ("high", "low"): {
        "name": "Cautious Affluent",
        "tagline": "High income, low spending",
        "description": "They can afford to spend but choose not to. This is the biggest untapped opportunity.",
        "strategy": [
            "Quality and value-for-money messaging",
            "Personalised offers based on past purchases",
            "Survey to understand why they hold back",
        ],
    },
    ("low", "high"): {
        "name": "Impulse Spenders",
        "tagline": "Low income, high spending",
        "description": "They spend more than their income suggests and respond strongly to trends.",
        "strategy": [
            "Trend-led campaigns and social media promotions",
            "Limited-time and flash offers",
            "Student or youth loyalty cards",
        ],
    },
    ("low", "low"): {
        "name": "Budget Conscious",
        "tagline": "Low income, low spending",
        "description": "Price-sensitive shoppers who buy mainly essentials.",
        "strategy": [
            "Discounts, bundles and clearance events",
            "Highlight affordable stores and essentials",
            "Low-cost retention, such as points on every purchase",
        ],
    },
    ("mid", "mid"): {
        "name": "Mainstream Shoppers",
        "tagline": "Average income, average spending",
        "description": "The largest, steadiest group. Reliable but easy to lose to competitors.",
        "strategy": [
            "Standard loyalty programme and regular newsletters",
            "Cross-sell related categories",
            "Seasonal campaigns to lift visit frequency",
        ],
    },
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns=COLUMN_NAMES)


def fit(df: pd.DataFrame, k: int = K):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    return scaler, model, X


def _level(z: float) -> str:
    if z > 0.5:
        return "high"
    if z < -0.5:
        return "low"
    return "mid"


def persona_keys(scaler: StandardScaler, model: KMeans) -> list[tuple[str, str]]:
    """Map each cluster to an (income level, spending level) persona key."""
    keys = []
    for z_income, z_spend in model.cluster_centers_:
        key = (_level(z_income), _level(z_spend))
        # a cluster close to the middle on one axis is treated as mainstream
        keys.append(key if key in PERSONAS else ("mid", "mid"))
    return keys


def elbow_and_silhouette(X: np.ndarray, k_max: int = 10) -> dict:
    wcss, sil = [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        wcss.append(round(float(km.inertia_), 2))
        if k >= 2:
            sil.append(round(float(silhouette_score(X, km.labels_)), 4))
    return {"k": list(range(1, k_max + 1)), "wcss": wcss,
            "silhouette_k": list(range(2, k_max + 1)), "silhouette": sil}


def build_export(df: pd.DataFrame, scaler, model, X) -> dict:
    labels = model.labels_
    keys = persona_keys(scaler, model)
    centers = scaler.inverse_transform(model.cluster_centers_)

    segments = []
    for c in range(model.n_clusters):
        part = df[labels == c]
        persona = PERSONAS[keys[c]]
        segments.append({
            "id": c,
            **persona,
            "size": int(len(part)),
            "share": round(len(part) / len(df), 4),
            "center": {"income": round(float(centers[c][0]), 2),
                       "spending": round(float(centers[c][1]), 2)},
            "avg_age": round(float(part["age"].mean()), 1),
            "avg_income": round(float(part["income"].mean()), 1),
            "avg_spending": round(float(part["spending"].mean()), 1),
            "income_range": [int(part["income"].min()), int(part["income"].max())],
            "spending_range": [int(part["spending"].min()), int(part["spending"].max())],
            "female_share": round(float((part["gender"] == "Female").mean()), 3),
        })

    return {
        "model": {
            "algorithm": "KMeans",
            "k": int(model.n_clusters),
            "features": FEATURES,
            "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
            "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
            "centroids_scaled": [[round(float(v), 6) for v in c] for c in model.cluster_centers_],
            "silhouette": round(float(silhouette_score(X, labels)), 4),
            # shared within-cluster variance, used for soft (probabilistic) membership
            "sigma2": round(float(soft_variance(model, X)), 6),
        },
        "dataset": {
            "rows": int(len(df)),
            "income_range": [int(df["income"].min()), int(df["income"].max())],
            "spending_range": [int(df["spending"].min()), int(df["spending"].max())],
            "age_range": [int(df["age"].min()), int(df["age"].max())],
        },
        "segments": segments,
        "customers": [
            {"id": int(r.customer_id), "gender": r.gender, "age": int(r.age),
             "income": int(r.income), "spending": int(r.spending), "segment": int(l)}
            for r, l in zip(df.itertuples(), labels)
        ],
        "diagnostics": elbow_and_silhouette(X),
    }


def soft_variance(model: KMeans, X: np.ndarray) -> float:
    """Variance per feature of the points around their own centre.

    K-Means is the limiting case of a Gaussian mixture where every cluster
    shares one spherical variance. Using that variance gives membership
    probabilities whose most likely segment always equals the K-Means label.
    """
    return model.inertia_ / (X.shape[0] * X.shape[1])


def segment_probabilities(income: float, spending: float, scaler, model, sigma2: float) -> np.ndarray:
    point = pd.DataFrame([[income, spending]], columns=FEATURES)
    d2 = ((model.cluster_centers_ - scaler.transform(point)[0]) ** 2).sum(axis=1)
    logits = -d2 / (2 * sigma2)
    p = np.exp(logits - logits.max())
    return p / p.sum()


def predict_segment(income: float, spending: float, scaler, model) -> int:
    point = pd.DataFrame([[income, spending]], columns=FEATURES)
    return int(model.predict(scaler.transform(point))[0])


def main() -> None:
    df = load_data()
    scaler, model, X = fit(df)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, ROOT / "models" / "kmeans.joblib")

    out = ROOT / "docs" / "data" / "model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    export = build_export(df, scaler, model, X)
    out.write_text(json.dumps(export, indent=1))
    # same data as a script, so docs/index.html also works when opened from disk (file://)
    out.with_suffix(".js").write_text("window.SEGMENTATION_MODEL = " + json.dumps(export) + ";\n")

    print(f"Silhouette score: {export['model']['silhouette']}")
    for s in export["segments"]:
        print(f"  cluster {s['id']}: {s['name']:<20} n={s['size']:<3} "
              f"income={s['avg_income']:>5}  spending={s['avg_spending']:>5}")
    print(f"Wrote {out.relative_to(ROOT)} and {out.with_suffix('.js').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
