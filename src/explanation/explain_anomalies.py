import pandas as pd
import numpy as np

# Using simple relative paths
INPUT_DATA = "data/processed/severity_data.csv"
OUTPUT_DATA = "data/processed/explained_data.csv"

def generate_explanations():
    print("Loading severity data...")
    df = pd.read_csv(INPUT_DATA)
    
    #start with a default explanation for normal rows
    df["explanation"] = "System operating normally."
    
    #create a function that applies rules to a single row
    def explain_row(row):
        # If it's a normal row, do nothing
        if row["anomaly_label"] == 1:
            return "System operating normally."
            
        cpu = row["cpu_utilization"]
        rolling_mean = row["cpu_rolling_mean_5"]
        rolling_std = row["cpu_rolling_std_5"]
        
        #Rule 1 Absolute massive spike (Looking at our EDA, anything > 1.0 is huge)
        if cpu > 1.0:
            return f"Massive CPU Spike: CPU reached {cpu:.2f}."
            
        # Rule 2: Sudden surge relative to its recent past
        # (If current CPU is more than 2x the rolling average)
        if cpu > (rolling_mean * 2):
            return f"Sudden CPU Surge: Usage jumped to {cpu:.2f}, double the recent average."
            
        # Rule 3: Highly unstable/bouncing CPU
        if rolling_std > 0.2:
            return f"Erratic CPU Behavior: High volatility detected (std: {rolling_std:.2f})."
            
        # Fallback for anomalies that the Isolation Forest caught but our simple rules missed
        return "Abnormal pattern detected by ML model."

    print("Analyzing anomaly patterns...")
    #apply the function to every row in the dataset
    df["explanation"] = df.apply(explain_row, axis=1)
    
    # save the final data
    df.to_csv(OUTPUT_DATA, index=False)
    
    # Let's preview some of our explanations!
    print("\nSample Explanations for Detected Anomalies:")
    anomalies = df[df["anomaly_label"] == -1]
    
    # Print just the timestamp, severity, and the new explanation for the first 5 anomalies
    print(anomalies[["timestamp", "severity", "explanation"]].head(5))
    
    print(f"\nSuccess! Explained data saved to: {OUTPUT_DATA}")

if __name__ == "__main__":
    generate_explanations()