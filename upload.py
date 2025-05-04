import os
from huggingface_hub import hf_hub_download , login
print(login(token=os.getenv("HF_TOKEN")))
print(os.getenv("HF_TOKEN"))
