import os
import pandas as pd

RAW_DATA_PATH = "data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_DIR = "data/ingested"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ingested_data.csv")

def ingest_data():
    print(f"📥 Reading raw data from: {RAW_DATA_PATH}")
    
    try:
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read raw data: {e}")
        return

    print(f"✅ Raw data shape: {df.shape}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save a clean copy (optional: sample first 1M rows if too large)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"📁 Ingested data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    ingest_data()

