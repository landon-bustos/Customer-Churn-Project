import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    os.environ['KAGGLE_CONFIG_DIR'] = '/Users/lb/kaggle'
    
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs('data', exist_ok=True)
    
    dataset_name = "blastchar/telco-customer-churn"
    print(f"Downloading dataset from Kaggle: {dataset_name}")
    api.dataset_download_files(dataset_name, path='data', unzip=True)
    
    print("Dataset downloaded successfully to ./data directory")

if __name__ == "__main__":
    download_dataset()
