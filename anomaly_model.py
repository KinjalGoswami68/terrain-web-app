import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

print("🚀 Step 1: Generating Synthetic NURE Radiometric Grid (K, eU, eTh)...")
# Simulating a grid of 10,000 geological locations
np.random.seed(42)
n_samples = 10000

# Geographic bounds matching your Streamlit map
latitudes = np.random.uniform(34.105, 34.145, n_samples)
longitudes = np.random.uniform(-118.135, -118.085, n_samples)

# Background radiometric values
K = np.random.normal(2.0, 0.5, n_samples)    # Potassium (%)
eTh = np.random.normal(8.0, 2.0, n_samples)  # Equivalent Thorium (ppm)
eU = np.random.normal(2.5, 0.8, n_samples)   # Equivalent Uranium (ppm)

# Injecting 50 artificial "anomalies" (high Uranium, low Thorium)
anomaly_indices = np.random.choice(n_samples, 50, replace=False)
eU[anomaly_indices] += np.random.normal(5.0, 1.0, 50)
eTh[anomaly_indices] -= np.random.normal(2.0, 0.5, 50)

df = pd.DataFrame({'latitude': latitudes, 'longitude': longitudes, 'K': K, 'eTh': eTh, 'eU': eU})
# Prevent division by zero
df = df[(df['K'] > 0.1) & (df['eTh'] > 0.1)] 

print("🧪 Step 2: Engineering Geological Features (Ratios & Migration Index)...")
# Domain-specific ratios the founder asked for
df['eU_eTh_ratio'] = df['eU'] / df['eTh']
df['eU_K_ratio'] = df['eU'] / df['K']
df['eTh_K_ratio'] = df['eTh'] / df['K']

# Uranium Migration Index
background_ratio = df['eTh'].mean() / df['eU'].mean()
df['uranium_migration_index'] = df['eU'] - (df['eTh'] / background_ratio)

print("🤖 Step 3: Running Unsupervised Model (Isolation Forest)...")
# We feed the model our engineered features, NOT the raw data
features = ['eU_eTh_ratio', 'eU_K_ratio', 'eTh_K_ratio', 'uranium_migration_index']
X = df[features]

# Initialize Isolation Forest (contamination=0.01 means we expect 1% of data to be anomalies)
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly_label'] = model.fit_predict(X)
df['anomaly_score'] = model.decision_function(X) # Lower scores = more anomalous

# Filter only the predicted anomalies (-1 label in Isolation Forest)
predicted_anomalies = df[df['anomaly_label'] == -1].copy()

# Save our AI predictions to a new database
predicted_anomalies.to_csv("predicted_anomalies.csv", index=False)
print(f"✅ Success! Found {len(predicted_anomalies)} anomalies. Saved to 'predicted_anomalies.csv'.")

# Quick visual check
plt.figure(figsize=(8, 6))
plt.scatter(df['longitude'], df['latitude'], c='lightgray', s=1, label='Background Terrain')
plt.scatter(predicted_anomalies['longitude'], predicted_anomalies['latitude'], c='red', s=20, label='Predicted Anomalies')
plt.title("Isolation Forest Radiometric Anomalies")
plt.legend()
plt.show()