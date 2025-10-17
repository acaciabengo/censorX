from flask import Flask
from .routes import api_blueprint
from .models.load_model import ModelLoader
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim


def create_app():
    app = Flask(__name__)

    # Configuration settings can be added here
    app.config["DEBUG"] = True

    # load the model
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model()
    # if cuda is available, move the model to GPU else change to cpu
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    else:
        model = model.to(torch.device("cpu"))

    app.config["MODEL"] = model
    app.config["TOKENIZER"] = tokenizer

    app.register_blueprint(api_blueprint, url_prefix="/api/v1")

    return app
