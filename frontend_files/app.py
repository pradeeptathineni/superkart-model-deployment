

import io
import json
import os

import pandas as pd
import requests
import streamlit as st


# Docker uses the backend service name; an environment variable also supports Codespaces
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860").rstrip("/")

st.set_page_config(page_title="SuperKart Sales", layout="centered")
st.title("SuperKart System")
st.write("Enter the product and store details to estimate total sales revenue.")
st.caption("Estimates use the supplied dataset's revenue units and observation period.")

# Show training categories so the form does not silently send misspelled labels
with st.form("single_prediction"):
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox(
        "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
    )
    Product_Allocated_Area = st.number_input(
        "Product Allocated Area", min_value=0.0, max_value=1.0,
        value=0.027, step=0.001, format="%.3f"
    )
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Store_Size = st.selectbox("Store Size", ["Medium", "Small", "High"])
    Store_Location_City_Type = st.selectbox(
        "Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"]
    )
    Store_Type = st.selectbox(
        "Store Type", ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"]
    )
    Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
    Store_Age_Years = st.number_input("Store Age (2025 reference year)", min_value=0, value=16)
    Product_Type_Category = st.selectbox(
        "Product Type Category", ["Non Perishables", "Perishables"]
    )
    predict = st.form_submit_button("Predict", type="primary")

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
    "Product_Type_Category": Product_Type_Category,
}

if predict:
    # A timeout and a clear error keep the page usable when the backend is unavailable
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.ok:
            result = response.json()
            st.success(f"Predicted Product Store Sales Total: {result['Sales']:,.2f}")
            for warning in result.get("warnings", []):
                st.warning(warning)
        else:
            st.error(response.json().get("error", "The prediction could not be completed."))
    except requests.RequestException:
        st.error("The prediction service could not be reached. Please try again.")

st.subheader("Batch Prediction")
st.write("Upload a CSV with the same ten feature columns used by the single prediction form.")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.button("Predict for Batch", type="primary"):
    try:
        content = uploaded_file.getvalue()
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={"file": (uploaded_file.name, content, "text/csv")},
            timeout=30,
        )
        if response.ok:
            results = response.json()
            # Map each numbered result back to its original CSV row
            batch_results = pd.read_csv(io.BytesIO(content))
            # Put the prediction first so it is visible before horizontal scrolling
            batch_results.insert(0, "Predicted_Sales", [results[str(i)] for i in range(len(batch_results))])
            st.success(f"Predictions completed for {len(batch_results)} products.")
            for warning in json.loads(response.headers.get("X-SuperKart-Warnings", "[]")):
                st.warning(warning)
            st.dataframe(batch_results, use_container_width=True)
            st.download_button(
                "Download predictions", batch_results.to_csv(index=False).encode("utf-8"),
                file_name="SuperKart_Batch_Predictions.csv", mime="text/csv"
            )
        else:
            st.error(response.json().get("error", "The batch could not be processed."))
    except (requests.RequestException, ValueError):
        st.error("Check the CSV and the prediction service, then try again.")
