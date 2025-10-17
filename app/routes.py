from flask import Blueprint, current_app, request, jsonify
from app.models.load_model import ModelLoader
import torch

# Create a Blueprint called "api"
api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    data = data.get("data", "")
    # if data is an array , ok else convert to array
    if not isinstance(data, list):
        data = [data]

    model = current_app.config["MODEL"]
    tokenizer = current_app.config.get("TOKENIZER")
    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    probabilities = model(**inputs).logits.softmax(dim=-1)[:, 1].tolist()
    return jsonify({"result": probabilities}), 200


@api_blueprint.route("/reload_model", methods=["POST"])
def reload_model():
    model_loader = ModelLoader()
    model = model_loader.load_model()
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    else:
        model = model.to(torch.device("cpu"))
    return jsonify({"message": "Model reloaded successfully"})


@api_blueprint.route("/download_model", methods=["POST"])
def download_model():
    model_loader = ModelLoader()
    model_loader.download_model()
    return jsonify({"message": "Model downloaded successfully"})
