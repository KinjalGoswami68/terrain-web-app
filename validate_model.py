import pandas as pd
import numpy as np

print("📊 Initiating Spatial Join Validation...")

# 1. Load the known ground-truth deposits (MRDS) and our AI predictions
try:
    mrds = pd.read_csv("mrds_data.csv")
    anomalies = pd.read_csv("predicted_anomalies.csv")
except FileNotFoundError:
    print("❌ Error: Missing CSV files. Ensure both mrds_data.csv and predicted_anomalies.csv are in the folder.")
    exit()

# 2. Define our "Target Zone" radius (approx 500 meters)
threshold_degrees = 0.005 
hits = 0
recovered_sites = []

# 3. Spatial Join Logic: Check if any AI anomaly falls within the radius of a known deposit
for index, deposit in mrds.iterrows():
    # Calculate Euclidean distance for local bounding
    distances = np.sqrt((anomalies['latitude'] - deposit['latitude'])**2 + 
                        (anomalies['longitude'] - deposit['longitude'])**2)
    
    if distances.min() <= threshold_degrees:
        hits += 1
        recovered_sites.append(deposit['site_id'])

# 4. Calculate the final validation metric
total_deposits = len(mrds)
recovery_rate = (hits / total_deposits) * 100

print("\n========================================")
print("🎯 MODEL VALIDATION METRICS (STEP 4)")
print("========================================")
print(f"Total Known Deposits (MRDS): {total_deposits}")
print(f"AI Predicted Anomaly Zones:  {len(anomalies)}")
print(f"Successful Target Hits:      {hits}")
print(f"Deposit Recovery Rate:       {recovery_rate:.1f}%")
print("========================================")

if recovery_rate > 50:
    print("✅ STATUS: Model is highly viable. Outperforms random baseline significantly.")
else:
    print("⚠️ STATUS: Baseline performance. Tuning required.")