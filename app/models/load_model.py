import os
import torch
from google.cloud import storage
from transformers import DistilBertForSequenceClassification, AutoTokenizer


class ModelLoader:
    def __init__(self, model_path=None, model_name=None):
        self.model_name = (
            "quant_distil_bert_model.pth" if model_name is None else model_name
        )

        if model_path:
            self.model_path = os.path.join(os.getcwd(), model_path)
        else:
            self.model_path = os.path.join(os.getcwd(), "models")

        self.model = None

    def download_model(self):
        # connect to Google Cloud Storage and download the model file
        auth_file_path = os.path.join(os.getcwd(), "auth.json")
        storage_client = storage.Client.from_service_account_json(auth_file_path)
        bucket_name = "censorx_models"
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(self.model_name)
        local_model_path = os.path.join(self.model_path, self.model_name)
        blob.download_to_filename(local_model_path)
        print(f"Model downloaded to {local_model_path}")
        # reload the model after downloading
        self.load_model()

    def load_model(self):
        local_model_path = os.path.join(self.model_path, self.model_name)
        if not os.path.exists(local_model_path):
            print("Model file not found locally. Downloading...")
            self.download_model()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        state_dict = torch.load(local_model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tokenizer = tokenizer
        self.model.eval()
        print("Model loaded successfully")
        return self.model, self.tokenizer
