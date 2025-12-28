import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load data
df = pd.read_csv("data/raw/Fraud_Data.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isna().sum())

# Check duplicates
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)
if duplicates > 0:
    df = df.drop_duplicates()

# Class imbalance
print("\nClass Distribution:")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))

# Purchase value distribution
plt.figure(figsize=(6,4))
sns.histplot(df['purchase_value'], bins=50)
plt.title("Purchase Value Distribution")
plt.tight_layout()
plt.savefig("purchase_value_distribution.png")
plt.close()

# Purchase value vs fraud
plt.figure(figsize=(6,4))
sns.boxplot(x='class', y='purchase_value', data=df)
plt.title("Purchase Value by Fraud Label")
plt.tight_layout()
plt.savefig("purchase_value_by_class.png")
plt.close()

print("\nEDA plots saved successfully.")