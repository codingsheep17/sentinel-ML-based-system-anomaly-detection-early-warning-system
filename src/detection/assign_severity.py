import pandas as pd

#Using relative paths
INPUT_DATA = "data/processed/labeled_data.csv"
OUTPUT_DATA = "data/processed/severity_data.csv"

def assign_severity():
    print("Loading scored data...")
    df = pd.read_csv(INPUT_DATA)
    
    # Start by assuming everything is Normal
    df["severity"] = "Normal"
    
    # Filter out only the rows the model flagged as anomalies
    anomalies = df[df["anomaly_label"] == -1].copy()
    
    if len(anomalies) > 0:
        print(f"Calculating severity for {len(anomalies)} anomalies...")
        
        #Divide the anomalies into 4 buckets based on their score.
        # In Isolation Forest, LOWER negative scores are WORSE (more anomalous).
        # So the lowest 25% of scores get "Critical", the highest 25% get "Low".
        labels = ["Critical", "High", "Medium", "Low"]
        
        anomalies["severity"] = pd.qcut(anomalies["anomaly_score"], q=4, labels=labels)
        
        # 4. Update our main dataset with these new severity labels
        df.update(anomalies)
    
    # Save the updated data
    df.to_csv(OUTPUT_DATA, index=False)
    
    print("\nSeverity breakdown:")
    print(df["severity"].value_counts())
    print(f"\nSuccess! Severity data saved to: {OUTPUT_DATA}")

if __name__ == "__main__":
    assign_severity()