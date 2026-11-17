"""DMBI CA2: Customer Segmentation and Behavioural Profiling for Retail Marketing

Dataset: Kaggle - Customer Personality Analysis (marketing_campaign.csv)
Install: pip install pandas numpy matplotlib seaborn scikit-learn scipy
Run: python CA2_Customer_Segmentation.py --data marketing_campaign.csv

The script prints the preprocessing summary, evaluates K-Means and Ward
Agglomerative clustering for K=2..8, creates customer profiles and saves
visualisations in the ./CA2_Results folder.
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
FEATURES = ["Income", "TotalSpending", "TotalPurchases", "Recency",
            "MntWines", "NumWebPurchases", "TotalChildren", "Age"]
SPENDING_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
                 "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases",
                 "NumStorePurchases", "NumDealsPurchases"]


def main():
    parser = argparse.ArgumentParser(description="Retail customer segmentation using data mining")
    parser.add_argument("--data", required=True, help="Path to Kaggle marketing_campaign.csv")
    parser.add_argument("--out", default="CA2_Results", help="Output directory")
    args = parser.parse_args()
    data_path = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Download Kaggle Customer Personality Analysis CSV.")

    df = pd.read_csv(data_path, sep='\t' if data_path.suffix.lower() == '.txt' else ',')
    # Some copies of the Kaggle file are tab-separated even when renamed as CSV.
    if len(df.columns) == 1:
        df = pd.read_csv(data_path, sep='\t')
    print("\n--- DATASET OVERVIEW ---")
    print(f"Original shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    if "Income" not in df.columns:
        raise ValueError("Expected Kaggle Customer Personality Analysis columns; 'Income' was not found.")

    # Data cleaning: median imputation, constant-column removal and invalid-record filtering.
    income_median = df["Income"].median()
    df["Income"] = df["Income"].fillna(income_median)
    for col in ["Z_CostContact", "Z_Revenue"]:
        if col in df.columns and df[col].nunique(dropna=False) <= 1:
            df.drop(columns=col, inplace=True)
    before_filter = len(df)
    if "Year_Birth" in df.columns:
        df = df[df["Year_Birth"] > 1900]
    df = df[df["Income"] != 666666].copy()
    print("\n--- PREPROCESSING SUMMARY ---")
    print(f"Income median used for imputation: {income_median:.1f}")
    print(f"Records removed as invalid/outlier: {before_filter - len(df)}")

    # Feature engineering.
    df["Age"] = 2026 - df["Year_Birth"]
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
    df["TotalPurchases"] = df[PURCHASE_COLS].sum(axis=1)
    campaign_cols = [c for c in ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                                  "AcceptedCmp4", "AcceptedCmp5", "Response"] if c in df.columns]
    df["CampaignsAccepted"] = df[campaign_cols].sum(axis=1)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Final customer records: {len(df):,}")
    print(f"Clustering features ({len(FEATURES)}): {', '.join(FEATURES)}")

    # Model comparison across K=2..8.
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km_labels = km.fit_predict(X_scaled)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hc_labels = hc.fit_predict(X_scaled)
        rows.append({"K": k, "KMeans_Inertia": km.inertia_,
                     "KMeans_Silhouette": silhouette_score(X_scaled, km_labels),
                     "KMeans_Davies_Bouldin": davies_bouldin_score(X_scaled, km_labels),
                     "Hierarchical_Silhouette": silhouette_score(X_scaled, hc_labels),
                     "Hierarchical_Davies_Bouldin": davies_bouldin_score(X_scaled, hc_labels)})
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(out / "clustering_evaluation.csv", index=False)
    print("\n--- CLUSTERING EVALUATION (K=2..8) ---")
    print(evaluation.round(4).to_string(index=False))

    # K=3 selected for business interpretability, as described in the CA2 report.
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    hier = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)
    print(f"\nAdjusted Rand Index (K=3 methods): {adjusted_rand_score(df['Cluster'], hier):.4f}")

    profile = df.groupby("Cluster").agg(
        Customers=("Cluster", "size"), Income=("Income", "mean"),
        TotalSpending=("TotalSpending", "mean"), TotalPurchases=("TotalPurchases", "mean"),
        Deals=("NumDealsPurchases", "mean"), WebVisits=("NumWebVisitsMonth", "mean"),
        TotalChildren=("TotalChildren", "mean"), Age=("Age", "mean"),
        CampaignsAccepted=("CampaignsAccepted", "mean")).round(2)
    profile["Customer_Percent"] = (profile["Customers"] / len(df) * 100).round(1)
    # Assign readable names based on cluster spending and deal purchasing behaviour.
    spending_rank = profile["TotalSpending"].rank(ascending=False, method="first")
    deal_rank = profile["Deals"].rank(ascending=False, method="first")
    names = {}
    high_value = spending_rank.idxmin()
    low_value = spending_rank.idxmax()
    remaining = [i for i in profile.index if i not in [high_value, low_value]]
    names[high_value] = "High-Value"
    names[low_value] = "Low-Engagement"
    names[remaining[0]] = "Discount-Oriented"
    profile.insert(0, "Segment", [names[i] for i in profile.index])
    profile.to_csv(out / "customer_segment_profiles.csv")
    df.to_csv(out / "segmented_customers.csv", index=False)
    print("\n--- CUSTOMER SEGMENT PROFILES ---")
    print(profile.to_string())

    # Visualisation 1: EDA distribution and spending categories.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    df[SPENDING_COLS].sum().sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Total Customer Spending by Product Category")
    ax.set_ylabel("Total Spending")
    ax.set_xlabel("Product Category")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(out / "01_spending_by_category.png", dpi=180); plt.close()

    # Visualisation 2: evaluation metrics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(evaluation.K, evaluation.KMeans_Silhouette, marker="o", label="K-Means")
    axes[0].plot(evaluation.K, evaluation.Hierarchical_Silhouette, marker="s", label="Hierarchical")
    axes[0].set_title("Silhouette Score by Number of Clusters"); axes[0].set_xlabel("K"); axes[0].legend()
    axes[1].plot(evaluation.K, evaluation.KMeans_Inertia, marker="o")
    axes[1].set_title("K-Means Inertia (Elbow Method)"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Inertia")
    plt.tight_layout(); plt.savefig(out / "02_clustering_evaluation.png", dpi=180); plt.close()

    # Visualisation 3: PCA projection of K-Means segments.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(components[:, 0], components[:, 1], c=df["Cluster"], cmap="viridis", s=16, alpha=.65)
    ax.set_title(f"Customer Segments using PCA (Explained variance: {pca.explained_variance_ratio_.sum():.1%})")
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout(); plt.savefig(out / "03_customer_segments_pca.png", dpi=180); plt.close()

    # Visualisation 4: mean feature profile heatmap.
    heat = df.groupby("Cluster")[FEATURES].mean()
    heat = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Standardised Customer Segment Profiles")
    plt.tight_layout(); plt.savefig(out / "04_segment_profile_heatmap.png", dpi=180); plt.close()

    # Visualisation 5: spending contribution by segment.
    contribution = df.groupby("Cluster")["TotalSpending"].sum()
    contribution.index = [names[i] for i in contribution.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    contribution.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Total Spending Contribution by Customer Segment")
    ax.set_ylabel("Total Spending"); ax.set_xlabel("Customer Segment")
    plt.tight_layout(); plt.savefig(out / "05_segment_spending_contribution.png", dpi=180); plt.close()

    # Optional dendrogram from a reproducible sample to keep memory/time manageable.
    sample = X_scaled[:min(500, len(X_scaled))]
    Z = linkage(sample, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=25, leaf_rotation=45, leaf_font_size=8, ax=ax)
    ax.set_title("Ward Hierarchical Clustering Dendrogram (Sample)")
    ax.set_xlabel("Merged observations / clusters"); ax.set_ylabel("Ward distance")
    plt.tight_layout(); plt.savefig(out / "06_ward_dendrogram.png", dpi=180); plt.close()

    print(f"\nVisualisations and result files saved in: {out.resolve()}")
    print("Business interpretation:")
    for _, r in profile.iterrows():
        print(f"- {r['Segment']}: {int(r['Customers'])} customers ({r['Customer_Percent']}%); "
              f"average spending={r['TotalSpending']:.2f}, average purchases={r['TotalPurchases']:.2f}.")
    print("Use high-value segments for retention and loyalty, discount-oriented customers for targeted offers, "
          "and low-engagement customers for carefully tested re-engagement campaigns.")


if __name__ == "__main__":
    main()
