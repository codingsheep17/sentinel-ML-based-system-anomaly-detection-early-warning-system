import urllib.request
from pathlib import Path

# 1. Define project root using pathlib (works on any OS)
PROJECT_ROOT = Path(__file__).resolve().parent

# 2. Define our folder structure
FOLDERS = [
    "app",
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src/data",
    "src/preprocessing",
    "src/features",
    "src/detection",
    "src/explanation",
    "src/utils",
    "scripts",
    "tests",
    "assets"
]

def setup():
    print("Creating project directories...")
    # 3. Create the folders
    for folder in FOLDERS:
        folder_path = PROJECT_ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Add a .gitkeep file so Git tracks these empty folders
        gitkeep_path = folder_path / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()

    print("Directories created successfully!")

    # 4. Download the dataset
    print("Downloading NAB AWS Cloudwatch Dataset...")
    # This is an authentic AWS EC2 CPU telemetry file from Numenta
    DATASET_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ec2_cpu_utilization.csv"

    # Download if it doesn't exist yet
    if not RAW_DATA_PATH.exists():
        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
        print(f"Dataset downloaded to: {RAW_DATA_PATH}")
    else:
        print("Dataset already exists!")

if __name__ == "__main__":
    setup()