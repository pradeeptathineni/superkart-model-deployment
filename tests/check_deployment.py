# Compare the running API with the saved model and check invalid requests

# Read addresses and save the check results
import json
import os
from pathlib import Path

# Load the same model and data used by the deployed API
import joblib
import numpy as np
import pandas as pd
import requests

# Locate the project and choose the API to check
root = Path(__file__).resolve().parents[1]
url = os.environ.get("MODEL_ROOT_URL", "http://127.0.0.1:7860").rstrip("/")

# Load one example and the full ten-row batch
model = joblib.load(root / "backend_files/superkart_model.joblib")
batch = pd.read_csv(root / "Batch_Data_SuperKart.csv")
payload = batch.iloc[0].to_dict()
checks = []

# Stop on a failed check and record each passing check
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)
    print("PASS:", name)

# Confirm that the server has loaded its model before sending predictions
health = requests.get(url + "/health", timeout=30)
check("model health", health.status_code == 200 and health.json()["status"] == "ok")

# Compare a single API estimate with the saved pipeline
single = requests.post(url + "/v1/predict", json=payload, timeout=30)
check("single request", single.status_code == 200)
np.testing.assert_allclose(single.json()["Sales"], model.predict(pd.DataFrame([payload]))[0], rtol=1e-6)
check("single prediction matches saved pipeline", True)

# Upload the CSV through the same multipart field used by the page
response = requests.post(
    url + "/v1/predictbatch",
    files={"file": ("batch.csv", batch.to_csv(index=False).encode(), "text/csv")},
    timeout=30,
)
check("batch request", response.status_code == 200)
check("all ten batch rows returned", set(response.json()) == {str(i) for i in range(len(batch))})

# Check both the numbered row order and the two-decimal predictions
actual = np.array([response.json()[str(i)] for i in range(len(batch))])
np.testing.assert_allclose(actual, model.predict(batch), rtol=1e-6, atol=0.011)
check("batch predictions preserve row order and match saved pipeline", True)
check("unseen batch category reported", "Supermarket Type3" in response.headers.get("X-SuperKart-Warnings", ""))

# Exercise the missing-field, range, type, and JSON-shape rejection paths
bad_requests = {
    "missing column": {"Product_MRP": 100},
    "negative weight": {**payload, "Product_Weight": -1},
    "invalid area": {**payload, "Product_Allocated_Area": 2},
    "invalid number": {**payload, "Product_MRP": "not a number"},
    "empty category": {**payload, "Store_Size": ""},
    "fractional store age": {**payload, "Store_Age_Years": 1.5},
    "wrong JSON shape": [payload],
}

# Every invalid example should return a readable 400 response
for label, data in bad_requests.items():
    invalid = requests.post(url + "/v1/predict", json=data, timeout=30)
    check(label + " rejected", invalid.status_code == 400 and "error" in invalid.json())

# Check missing and empty CSV uploads separately
empty_upload = requests.post(url + "/v1/predictbatch", timeout=30)
check("missing upload rejected", empty_upload.status_code == 400)
empty_csv = requests.post(
    url + "/v1/predictbatch",
    files={"file": batch.iloc[:0].to_csv(index=False).encode()}, timeout=30,
)
check("empty CSV rejected", empty_csv.status_code == 400)

# Unknown categories can be estimated, but the response must make that limitation visible
unknown = requests.post(
    url + "/v1/predict", json={**payload, "Store_Type": "Supermarket Type3"}, timeout=30
)
check("unknown category is visible", unknown.status_code == 200 and bool(unknown.json().get("warnings")))

# Save the endpoint and completed checks for the deployment record
evidence = {"backend_url": url, "passed": len(checks), "checks": checks}
(root / "results").mkdir(exist_ok=True)
(root / "results/deployment_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(f"Completed {len(checks)} checks against {url}.")
