# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Read model settings and locate files independently of the working directory
import json
from pathlib import Path

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Reject oversized uploads before reading their contents into memory
superkart_api.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Load the trained model
# CHANGED: resolve the path from this file so launch location does not affect loading
# model = joblib.load("superkart_model.joblib")
model_dir = Path(__file__).resolve().parent
model = joblib.load(model_dir / "superkart_model.joblib")

# Use the saved feature names and categories to check incoming records
metadata = json.loads((model_dir / "model_metadata.json").read_text())
features = metadata["features"]
numeric_features = metadata["numeric_features"]
categories = metadata["categories"]

# Check required fields, numeric values, and categories before prediction
def prepare_input(frame):
    # Stop early when columns or rows are missing, or the batch is too large
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Provide at least one product row.")
    if len(frame) > 5000:
        raise ValueError("A batch can contain at most 5,000 rows.")

    # Keep the training column order and combine the two Regular sugar labels
    frame = frame.loc[:, features].copy()
    frame["Product_Sugar_Content"] = frame["Product_Sugar_Content"].replace({"reg": "Regular"})

    # Convert numeric fields and reject missing, infinite, or negative values
    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numbers = frame[numeric_features].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("Numeric inputs must contain finite, nonnegative values.")

    # Display area is a ratio, and store age must use whole years
    if (frame["Product_Allocated_Area"] > 1).any():
        raise ValueError("Product_Allocated_Area must be between 0 and 1.")
    if (frame["Store_Age_Years"] % 1 != 0).any():
        raise ValueError("Store_Age_Years must be a whole number.")

    # Report unfamiliar categories without treating them as validated examples
    warnings = []
    for column, known_values in categories.items():
        if frame[column].isna().any() or not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"{column} must contain nonempty text.")
        unknown = sorted(set(frame[column]) - set(known_values))
        if unknown:
            warnings.append(f"{column} was not seen in training: {', '.join(unknown)}")

    # Return the checked inputs together with any warnings for the caller
    return frame, warnings

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Let Docker check that the API and its model have loaded
@superkart_api.get('/health')
def health():
    return jsonify({"status": "ok", "model": metadata["model_name"]})

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    # CHANGED: malformed or incomplete input should return 400 rather than a server error
    # data = request.get_json()
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send one product as a JSON object."}), 400

    # Check column names before extracting values from the request
    missing = [column for column in features if column not in data]
    if missing:
        return jsonify({"error": "Missing columns: " + ", ".join(missing)}), 400

    # Extract relevant features from the input data
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Validate the values, with a clear response if the request cannot be used
    try:
        input_data, warnings = prepare_input(input_data)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    # CHANGED: include a warning only when an input category was absent from training
    # return jsonify({'Sales': prediction})
    result = {'Sales': prediction}
    if warnings:
        result['warnings'] = warnings
    return jsonify(result)

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    # CHANGED: report a missing upload before trying to read its file field
    if 'file' not in request.files:
        return jsonify({"error": "Upload a CSV using the file field."}), 400
    file = request.files['file']

    # Read the file into a DataFrame
    # CHANGED: reject unreadable CSVs and invalid rows before calling the model
    try:
        input_data = pd.read_csv(file)
        input_data, warnings = prepare_input(input_data)
    except (ValueError, TypeError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400

    # Make predictions for the batch data
    predictions = model.predict(input_data).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    # Return the same numbered values, with category warnings kept in a response header
    # CHANGED: a response object allows warnings without changing the prediction dictionary
    # return output_dict
    response = jsonify(output_dict)
    if warnings:
        response.headers['X-SuperKart-Warnings'] = json.dumps(warnings)
    return response

# Run the Flask app
# CHANGED: disable debug mode and bind port 7860 for local checks; Docker uses Gunicorn
# superkart_api.run(debug=True)
if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=7860, debug=False)
