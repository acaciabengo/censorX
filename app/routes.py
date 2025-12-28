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
        logits = model(**inputs).logits
        probs = logits.softmax(dim=-1).tolist()  # shape: (batch_size, num_classes)
    # Map class indices to names
    label_names = [labels[i] for i in range(len(labels))]
    results = [dict(zip(label_names, p)) for p in probs]
    # Return a single dict if only one input, else a list
    if len(results) == 1:
        return jsonify(results[0]), 200
    else:
        return jsonify(results), 200


@api_blueprint.route("/image", methods=["POST"])
def predict_image():
    import requests

    labels = {"drawings": 0, "hentai": 1, "neutral": 2, "porn": 3, "sexy": 4}
    id_to_label = {v: k for k, v in labels.items()}

    image = None

    # Try file upload first
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Failed to read uploaded image: {str(e)}"}), 400
    else:
        # Try URL in JSON
        data = request.json
        url = data.get("url") if data else None
        if url:
            try:
                response = requests.get(url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400
        else:
            return jsonify({"error": "No file or URL provided"}), 400

    try:
        model = current_app.config["IMAGE_MODEL"]
        processor = current_app.config["IMAGE_TRANSFORM"]

        inputs = processor(images=image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1).tolist()
        label_names = [id_to_label[i] for i in range(len(id_to_label))]
        results = [dict(zip(label_names, p)) for p in probs]
        if len(results) == 1:
            return jsonify(results[0]), 200
        else:
            return jsonify(results), 200
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
