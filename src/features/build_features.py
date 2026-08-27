import pandas as pd

INPUT_DATA = "D:/Desktop/sentinelml/data/processed/processed.csv"
OUTPUT_DATA = "D:/Desktop/sentinelml/data/processed/processed_features.csv"

def build_features():
    print("Loading cleaned dataset...")
    #converting to datetime again
    df = pd.read_csv(INPUT_DATA, parse_dates=["timestamp"])

    print("Adding time-based features...")
    # Extract the hour (0-23) and day of the week (0=Monday, 6=Sunday)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    print("Adding rolling window features...")
    # Rolling mean Smooths out the data to show the general trend over the last 5 readings
    df["cpu_rolling_mean_5"] = df["cpu_utilization"].rolling(window=5).mean()
    
    # Rolling std Measures how much the CPU usage is jumping around over the last 5 readings
    df["cpu_rolling_std_5"] = df["cpu_utilization"].rolling(window=5).std()

    print("Handling missing values from rolling windows...")
    # The first 4 rows won't have enough history to calculate a 5-window rolling average, 
    df = df.bfill()

    # Save the new dataset with our engineered features
    df.to_csv(OUTPUT_DATA, index=False)
    print(f"Success! Featured data saved to: {OUTPUT_DATA}")
    
    # Display the first few rows to verify
    print("\nSneak peek of our new features:")
    print(df.head(5))

if __name__ == "__main__":
    build_features()