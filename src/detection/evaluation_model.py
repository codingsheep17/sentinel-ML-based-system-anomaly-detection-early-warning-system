import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Using simple relative paths
INPUT_DATA = "data/processed/explained_data.csv"

def evaluate():
    print("Loading data for evaluation...")
    df = pd.read_csv(INPUT_DATA)
    
    #Create our "Mock Ground Truth" (Pretending an expert labeled this)
    #if CPU > 0.85, it's a true anomaly (-1). Otherwise, normal (1).
    def mock_expert(cpu):
        if cpu > 0.85:
            return -1
        return 1
        
    df["true_label"] = df["cpu_utilization"].apply(mock_expert)
    
    print("\n--- Confusion Matrix ---")
    #generate the Confusion Matrix
    #this compares the expert's true labels against our model's predictions (anomaly_label)
    cm = confusion_matrix(df["true_label"], df["anomaly_label"], labels=[1, -1])
    
    print(f"True Positives (Normal correctly identified as Normal): {cm[0][0]}")
    print(f"False Positives (Normal falsely flagged as Anomaly): {cm[0][1]}")
    print(f"False Negatives (Real Anomaly missed by the model): {cm[1][0]}")
    print(f"True Negatives (Real Anomaly correctly caught!): {cm[1][1]}")
    
    print("\n--- Classification Report (Precision & Recall) ---")
    #generate Precision, Recall, and F1-Score
    report = classification_report(df["true_label"], df["anomaly_label"], 
                                   target_names=["Anomaly (-1)", "Normal (1)"])
    print(report)

if __name__ == "__main__":
    evaluate()