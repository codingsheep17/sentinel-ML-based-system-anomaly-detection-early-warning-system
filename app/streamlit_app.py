import streamlit as st
import pandas as pd
from pathlib import Path

#Setup for path
# __file__ looks at this exact script
# .parent goes up to the 'app' folder 
# .parent again goes up to the 'sentinelml' root folder.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "explained_data.csv"

st.set_page_config(page_title="SentinelML Dashboard", page_icon="🛡️", layout="wide")

#DATA LOADING
@st.cache_data
def load_data():
    # Load the data and ensure timestamp is treated as a date
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df

def main():
    # --- HEADER ---
    st.title("🛡️ SentinelML: Early Warning System")
    st.markdown("Monitoring AWS EC2 CPU Telemetry")

    # Load data using our cached function
    df = load_data()

    # --- KPI CALCULATIONS ---
    total_records = len(df)
    anomalies_df = df[df["anomaly_label"] == -1]
    total_anomalies = len(anomalies_df)
    critical_anomalies = len(anomalies_df[anomalies_df["severity"] == "Critical"])

    # --- TOP ROW: METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Telemetry Records", total_records)
    with col2:
        st.metric("Detected Anomalies", total_anomalies)
    with col3:
        st.metric("Critical Alerts", critical_anomalies)

    st.divider()

    #MIDDLE ROW: CHART
    st.subheader("📈 System CPU Utilization Over Time")
    # Streamlit line charts need the time to be the "index" (the main label for the row)
    chart_data = df.set_index("timestamp")[["cpu_utilization"]]
    st.line_chart(chart_data)

    st.divider()

    #BOTTOM ROW: ALERT LOG
    st.subheader("🚨 Recent Anomalies Log")
    if total_anomalies > 0:
        # Pick only the useful columns and put the newest alerts at the top
        display_cols = ["timestamp", "cpu_utilization", "severity", "explanation"]
        recent_anomalies = anomalies_df[display_cols].sort_values("timestamp", ascending=False)
        
        # Draw the table on the screen
        st.dataframe(recent_anomalies, use_container_width=True)
    else:
        st.success("No anomalies detected! System is operating normally.")

if __name__ == "__main__":
    main()