import pandas as pd

# Load data
df = pd.read_csv("data/raw/Fraud_Data.csv")

# Convert timestamps
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])

# Time since signup (hours)
df['time_since_signup'] = (
    df['purchase_time'] - df['signup_time']
).dt.total_seconds() / 3600

# Time-based features
df['hour_of_day'] = df['purchase_time'].dt.hour
df['day_of_week'] = df['purchase_time'].dt.dayofweek

# Convert IP to integer
df['ip_int'] = df['ip_address'].astype(int)

# Load IP-country mapping
ip_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")

# NOTE: Range-based merge will be handled in modeling phase
# For now, we document the logic conceptually

# Save processed data
df.to_csv("data/processed/fraud_features.csv", index=False)

print("Feature engineering completed.")