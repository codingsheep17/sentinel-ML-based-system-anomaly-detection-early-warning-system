import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Using simple relative paths
INPUT_DATA = "D:/Desktop/sentinelml/data/processed/processed_features.csv"
OUTPUT_DATA = "D:/Desktop/sentinelml/data/processed/labeled_data.csv"

def train_and_score():
    print("Loading feature data...")
    # Read the data, keeping the timestamp intact
    df = pd.read_csv(INPUT_DATA, parse_dates=["timestamp"])
    
    #Select the columns the model will actually look at. 
    # We do NOT pass the timestamp string to the model.
    features = ["cpu_utilization", "hour", "day_of_week", "cpu_rolling_mean_5", "cpu_rolling_std_5"]
    X = df[features]
    
    print("Initializing Isolation Forest...")
    #Set up the model. 
    # contamination=0.01 means we assume roughly 1% of our dataset contains anomalies.
    # random_state=42 ensures we get the exact same results every time we run it.
    model = IsolationForest(contamination=0.01, random_state=42)
    
    print("Training the model and making predictions...")
    #Fit the model to our data and get the strict labels in one step
    # Returns 1 for normal, -1 for anomaly
    df["anomaly_label"] = model.fit_predict(X)
    
    # 4. Get the raw anomaly score 
    # In scikit-learn, lower/negative numbers mean MORE anomalous.
    df["anomaly_score"] = model.decision_function(X) #tells about the model confidence about prediction
    
    print("Saving scored data...")
    df.to_csv(OUTPUT_DATA, index=False)
    
    # Let's see how many anomalies it found!
    anomalies_count = len(df[df["anomaly_label"] == -1])
    total_rows = len(df)
    
    print(f"Success! Evaluated {total_rows} records.")
    print(f"The model flagged {anomalies_count} potential anomalies.")
    print(f"Scored data saved to: {OUTPUT_DATA}")
    
    print(f"Saving trained model artifact to {"models/isolation_forest.joblib"}...")
    joblib.dump(model, "models/isolation_forest.joblib")
    
    print("Saving scored data...")

if __name__ == "__main__":
    train_and_score()