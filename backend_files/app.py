
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


superkart_api = Flask("SuperKart")
superkart_api.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Load relative to this file so the API also works outside its Docker directory
model_dir = Path(__file__).resolve().parent
model = joblib.load(model_dir / "superkart_model.joblib")
metadata = json.loads((model_dir / "model_metadata.json").read_text())
features = metadata["features"]
numeric_features = metadata["numeric_features"]
categories = metadata["categories"]


def prepare_input(frame):
    """Check the input columns and report categories absent from training."""
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Provide at least one product row.")
    if len(frame) > 5000:
        raise ValueError("A batch can contain at most 5,000 rows.")

    # Keep the same feature order and sugar labels used during training
    frame = frame.loc[:, features].copy()
    frame["Product_Sugar_Content"] = frame["Product_Sugar_Content"].replace(
        {"reg": "Regular"}
    )
    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numbers = frame[numeric_features].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("Numeric inputs must contain finite, nonnegative values.")
    if (frame["Product_Allocated_Area"] > 1).any():
        raise ValueError("Product_Allocated_Area must be between 0 and 1.")
    if (frame["Store_Age_Years"] % 1 != 0).any():
        raise ValueError("Store_Age_Years must be a whole number.")

    warnings = []
    for column, known_values in categories.items():
        if frame[column].isna().any() or not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"{column} must contain nonempty text.")
        unknown = sorted(set(frame[column]) - set(known_values))
        if unknown:
            warnings.append(f"{column} was not seen in training: {', '.join(unknown)}")
    return frame, warnings


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok", "model": metadata["model_name"]})


@superkart_api.post("/v1/predict")
def predict_sales():
    # Validate requests before prediction instead of returning a server error
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send one product as a JSON object."}), 400
    try:
        input_data, warnings = prepare_input(pd.DataFrame([payload]))
        prediction = float(model.predict(input_data)[0])
        result = {"Sales": prediction}
        if warnings:
            result["warnings"] = warnings
        return jsonify(result)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV using the file field."}), 400
    try:
        input_data, warnings = prepare_input(pd.read_csv(request.files["file"]))
        predictions = model.predict(input_data).tolist()
        output = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
        response = jsonify(output)
        # Preserve the course's index-keyed response and put warnings in a header
        if warnings:
            response.headers["X-SuperKart-Warnings"] = json.dumps(warnings)
        return response
    except (ValueError, TypeError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    # Bind port 7860 for local checks; the Docker image uses Gunicorn
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
