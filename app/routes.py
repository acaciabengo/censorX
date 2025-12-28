from flask import Blueprint, current_app, request, jsonify
from app.models.load_model import ModelLoader, TEXT_MODEL_NAME, IMAGE_MODEL_NAME
import torch
from PIL import Image
import io

# Create a Blueprint called "api"
api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/text", methods=["POST"])
def predict_text():
    labels = {0: "Not_NSFW", 1: "NSFW"}

    data = request.json
    data = data.get("data", "")
    # if data is an array , ok else convert to array
    if not isinstance(data, list):
        data = [data]

    model = current_app.config["TEXT_MODEL"]
    tokenizer = current_app.config.get("TEXT_TOKENIZER")
    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probabilities = model(**inputs).logits.softmax(dim=-1)[:, 1].tolist()
        labels_probabilities = {labels[i]: prob for i, prob in enumerate(probabilities)}
    return jsonify({"result": labels_probabilities}), 200


@api_blueprint.route("/image", methods=["POST"])
def predict_image():
    labels = {"drawings": 0, "hentai": 1, "neutral": 2, "porn": 3, "sexy": 4}
    id_to_label = {v: k for k, v in labels.items()}

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        model = current_app.config["IMAGE_MODEL"]
        processor = current_app.config["IMAGE_TRANSFORM"]

        inputs = processor(images=image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = outputs.logits.softmax(dim=-1)[:, 1].tolist()
            # convert to labels
            labels_probabilities = {
                id_to_label[i]: prob for i, prob in enumerate(probabilities)
            }

        return jsonify({"result": labels_probabilities}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_blueprint.route("/reload_model", methods=["POST"])
def reload_model():
    model_type = request.json.get("type", "all")  # text, image, or all
    model_loader = ModelLoader()
    device = next(
        current_app.config["TEXT_MODEL"].parameters()
    ).device  # Reuse current device

    if model_type in ["text", "all"]:
        text_model, text_tokenizer = model_loader.load_text_model()
        text_model.to(device)
        current_app.config["TEXT_MODEL"] = text_model
        current_app.config["TEXT_TOKENIZER"] = text_tokenizer

    if model_type in ["image", "all"]:
        image_model, image_transform = model_loader.load_image_model()
        image_model.to(device)
        current_app.config["IMAGE_MODEL"] = image_model
        current_app.config["IMAGE_TRANSFORM"] = image_transform

    return jsonify({"message": f"Models ({model_type}) reloaded successfully"})


@api_blueprint.route("/download_model", methods=["POST"])
def download_model():
    model_type = request.json.get("type", "all")
    model_loader = ModelLoader()

    try:
        if model_type in ["text", "all"]:
            model_loader.download_model(TEXT_MODEL_NAME)
        if model_type in ["image", "all"]:
            model_loader.download_model(IMAGE_MODEL_NAME)
        return jsonify({"message": "Models downloaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
