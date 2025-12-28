from flask import Flask
from .routes import api_blueprint
from .models.load_model import ModelLoader
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_app():
    app = Flask(__name__)

    # Configuration settings can be added here
    app.config["DEBUG"] = True

    # load the models
    model_loader = ModelLoader()

    # Check if we want to use quantization (e.g. if no GPU is available, or explicitly requested)
    # For this example, let's say we quantize if we are on CPU to speed it up.
    device = get_device()
    use_quantization = device.type == 'cpu' 
    
    # load the text model, a finetuned DistilBERT model for text classification
    text_model, text_tokenizer = model_loader.load_text_model(quantize=use_quantization)

    # load the image model, a finetuned ViT model for image classification
    image_model, image_transform = model_loader.load_image_model(quantize=use_quantization)

    # load the models into the GPU or CPU

    # if cuda is available, move the model to GPU else change to cpu
    print("Using device:", device)

    # Only move to device if NOT quantized (quantized models are usually CPU only in PyTorch dynamic quantization)
    if not use_quantization:
        text_model.to(device)
        image_model.to(device)

    app.config["TEXT_MODEL"] = text_model
    app.config["TEXT_TOKENIZER"] = text_tokenizer
    app.config["IMAGE_MODEL"] = image_model
    app.config["IMAGE_TRANSFORM"] = image_transform

    app.register_blueprint(api_blueprint, url_prefix="/api/v1")

    return app
