import os
import torch
from google.cloud import storage
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
    ViTForImageClassification,
    ViTImageProcessor,
)


from torchao.quantization import quantize_, int8_weight_only

TEXT_MODEL_NAME = "censorx_bert.pth"
IMAGE_MODEL_NAME = "censorx_vit.pth"


class ModelLoader:
    def __init__(self, model_path=None):
        if model_path:
            self.model_path = os.path.join(os.getcwd(), model_path)
        else:
            self.model_path = os.path.join(os.getcwd(), "models")

        # Ensure directory exists
        os.makedirs(self.model_path, exist_ok=True)

    def download_model(self, model_name):
        # connect to Google Cloud Storage and download the model file
        auth_file_path = os.path.join(os.getcwd(), "auth.json")
        try:
            storage_client = storage.Client.from_service_account_json(auth_file_path)
            bucket_name = "censorx_models"
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(model_name)
            local_model_path = os.path.join(self.model_path, model_name)
            blob.download_to_filename(local_model_path)
            print(f"Model {model_name} downloaded to {local_model_path}")
        except Exception as e:
            print(f"Failed to download model {model_name}: {e}")
            # In production you might want to raise, but for now we print
            raise e

    def load_text_model(self, quantize=False):
        local_model_path = os.path.join(self.model_path, TEXT_MODEL_NAME)
        if not os.path.exists(local_model_path):
            print(
                f"Text model file not found locally at {local_model_path}. Downloading..."
            )
            self.download_model(TEXT_MODEL_NAME)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        state_dict = torch.load(local_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        if quantize:
            print("Quantizing text model with torchao...")
            model = quantize_(
                model,
                int8_weight_only,
            )
            # compile for speed
            model = torch.compile(model, mode="max-autotune")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model.eval()
        print("Text Model loaded successfully")
        return model, tokenizer

    def load_image_model(self, quantize=False):
        local_model_path = os.path.join(self.model_path, IMAGE_MODEL_NAME)
        if not os.path.exists(local_model_path):
            print(
                f"Image model file not found locally at {local_model_path}. Downloading..."
            )
            self.download_model(IMAGE_MODEL_NAME)

        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=5
        )
        state_dict = torch.load(local_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        if quantize:
            print("Quantizing image model with torchao...")
            model = quantize_(
                model,
                int8_weight_only,
            )
            # compile for speed
            model = torch.compile(model, mode="max-autotune")

        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        model.eval()
        print("Image Model loaded successfully")
        return model, processor
