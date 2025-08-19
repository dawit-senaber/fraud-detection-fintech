import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Load data
fraud_data = pd.read_csv("./data/raw/Fraud_Data.csv")
ip_country = pd.read_csv("./data/raw/IpAddress_to_Country.csv")

print(f"Original dataset size: {len(fraud_data)} records")
print(f"Missing IPs before: {fraud_data['ip_address'].isna().sum()}")
fraud_data = fraud_data.dropna(subset=["ip_address"])
print(f"Missing IPs after: {fraud_data['ip_address'].isna().sum()}")


# Convert float-like IPs to integers
def convert_ip(ip):
    try:
        # Handle both float strings and integer strings
        return int(float(ip))
    except (ValueError, TypeError):
        return np.nan


print("Converting IPs to integers...")
fraud_data["ip_int"] = [convert_ip(ip) for ip in tqdm(fraud_data["ip_address"])]

# Check for invalid IPs
invalid_ips = fraud_data["ip_int"].isna().sum()
print(f"Invalid IP formats found: {invalid_ips}")

if invalid_ips > 0:
    print("Sample invalid IPs:")
    print(fraud_data.loc[fraud_data["ip_int"].isna(), "ip_address"].head(5))

# Filter out invalid IPs
fraud_data = fraud_data.dropna(subset=["ip_int"])
fraud_data["ip_int"] = fraud_data["ip_int"].astype(np.int64)

# Optimized country mapping
print("Merging IPs with country data...")
ip_country_sorted = ip_country.sort_values("lower_bound_ip_address").reset_index(
    drop=True
)

# Create a list for country assignment
country_list = [None] * len(fraud_data)

# Vectorized lookup using searchsorted
sorted_lowers = ip_country_sorted["lower_bound_ip_address"].values
sorted_uppers = ip_country_sorted["upper_bound_ip_address"].values
sorted_countries = ip_country_sorted["country"].values

# Find positions where IPs would be inserted in sorted_lowers
idxs = np.searchsorted(sorted_lowers, fraud_data["ip_int"].values, side="right") - 1

# Check if IP falls within the found interval


for i in tqdm(range(len(fraud_data))):
    idx = idxs[i]
    if idx >= 0 and sorted_lowers[idx] <= fraud_data["ip_int"].iloc[i] <= sorted_uppers[idx]:
        country_list[i] = sorted_countries[idx]

fraud_data["country"] = country_list


# Feature engineering
print("Engineering features...")
for col in ["purchase_time", "signup_time"]:
    fraud_data[col] = pd.to_datetime(fraud_data[col], errors="coerce")

fraud_data["time_since_signup"] = (
    fraud_data["purchase_time"] - fraud_data["signup_time"]
).dt.total_seconds()

fraud_data["purchase_hour"] = fraud_data["purchase_time"].dt.hour
fraud_data["purchase_day"] = fraud_data["purchase_time"].dt.dayofweek

# Cyclic encoding for time
fraud_data["hour_sin"] = np.sin(2 * np.pi * fraud_data["purchase_hour"] / 24)
fraud_data["hour_cos"] = np.cos(2 * np.pi * fraud_data["purchase_hour"] / 24)

# Handle other missing values
print(f"Missing values before cleaning: {fraud_data.isna().sum().sum()}")
fraud_data = fraud_data.dropna(
    subset=["device_id", "source", "browser", "country", "time_since_signup"]
)
print(f"Missing values after cleaning: {fraud_data.isna().sum().sum()}")

# Save processed data
fraud_data.to_csv("./data/processed/processed_ecommerce.csv", index=False)
print("\nE-commerce data processed successfully!")
print(f"Final dataset size: {len(fraud_data)} records")
print(f"Fraud rate: {fraud_data['class'].mean():.4f}")
print(f"Columns in final dataset: {list(fraud_data.columns)}")
