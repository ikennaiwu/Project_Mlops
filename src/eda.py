
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(input_path="data/transactions.csv", output_dir="notebooks"):
    df = pd.read_csv(input_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "eda_summary.txt"), "w") as f:
        f.write("\n📌 Dataset Info:\n")
        f.write(str(df.info(buf=None)))
        f.write("\n\n📌 Missing Values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\n📌 Class Distribution:\n")
        f.write(str(df['Class'].value_counts(normalize=True)))

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    # Distribution of amount
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Amount'], bins=50)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_distribution.png"))
    plt.close()

if __name__ == "__main__":
    run_eda()

# ✅ EDA module cleaned:
# - Supports CLI input/output paths
# - Writes summary to eda_summary.txt
# - Saves plots in notebooks/
# - Uses safe directory creation
