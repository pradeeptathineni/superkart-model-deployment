# Import the page, table, and API libraries
import streamlit as st
import pandas as pd
import requests

# Read uploaded CSV bytes, API warnings, and the deployment address
import io
import json
import os

# Base URL of the Flask backend
# CHANGED: allow a configured address while keeping the Docker service name as the default
# BACKEND_URL = "http://backend:7860"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860").rstrip('/')

# Page title
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

# Clarify the revenue units because the data does not identify a currency or quarter
st.caption("Estimates use the dataset's revenue units and observation period.")

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])

# CHANGED: keep all three decimal places and limit the display-area ratio to 0-1
# Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_Allocated_Area = st.number_input(
    "Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027,
    step=0.001, format="%.3f"
)

# Choose store labels with the same spelling used for training
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])

# CHANGED: remove Type3 from the form because no Type3 store was present in training
# Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Store_Type = st.selectbox(
    "Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
)

# Use the same product prefix and perishable category labels as the saved pipeline
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])

# CHANGED: state the fixed reference year so store age has the same meaning at prediction time
# Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Store_Age_Years = st.number_input("Store Age (2025 reference year)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):
    # CHANGED: stop waiting after 30 seconds and explain a connection failure on the page
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=product_data,
            timeout=30
        )

        # Show the returned estimate and any unfamiliar-category warnings
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]

            # CHANGED: omit the currency symbol because the data does not specify a currency
            # st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
            st.success(f"Predicted Product Store Sales Total: {predicted_sales:,.2f}")
            for warning in result.get("warnings", []):
                st.warning(warning)
        else:
            st.error(response.json().get("error", "Unable to connect to the prediction API."))
    except requests.RequestException:
        st.error("Unable to connect to the prediction API. Please try again.")

# Batch Prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

# Wait for an uploaded file before showing the batch prediction action
if uploaded_file is not None:
    if st.button("Predict for Batch", type='primary'):
        # Send the uploaded file and handle a backend that cannot be reached
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": uploaded_file},
                timeout=30
            )

            # Turn the numbered API results into a table that follows the CSV rows
            if response.status_code == 200:
                results = response.json()

                # CHANGED: give the row count so it can be checked against the uploaded file
                # st.success("Predictions completed successfully!")
                st.success(f"Predictions completed for {len(results)} products.")

                # Show category warnings without mixing them into the numbered predictions
                for warning in json.loads(response.headers.get("X-SuperKart-Warnings", "[]")):
                    st.warning(warning)

                # Check if all values are scalars
                # DEPRECATED: generic list/nested-dict branches; this API returns numbered scalar values
                # if all(not isinstance(v, (list, dict)) for v in results.values()):
                # df = pd.DataFrame([results])  # This puts every prediction across one row
                # CHANGED: attach each prediction to its input row and show sales first
                try:
                    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                    df.insert(0, "Predicted_Sales", [results[str(i)] for i in range(len(df))])
                    st.dataframe(df, use_container_width=True)

                    # Let the user save the inputs and their predictions together
                    st.download_button(
                        "Download predictions", df.to_csv(index=False).encode('utf-8'),
                        file_name="SuperKart_Batch_Predictions.csv", mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Unable to display results as a table: {e}")
                    st.json(results)
            else:
                st.error(response.json().get("error", "Unable to connect to the prediction API."))
        except requests.RequestException:
            st.error("Unable to connect to the prediction API. Please try again.")
