from huggingface_hub import HfApi
import os

# Get the Hugging Face token from environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the single model file
api.upload_file(
    path_or_fileobj="/home/pdp28/Projects/AI/Housing-prediction-Streamlit/model.pkl",  # Path to the .pkl file
    repo_id="damodaraprakash/house-price-predictor",  # Your repo on Hugging Face
    repo_type="model",  # Type of the repo (model)
    path_in_repo="model.pkl"  # The path where the file will be stored in the repo
)
