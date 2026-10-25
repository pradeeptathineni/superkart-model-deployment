"""Compare the running API with the saved model and check invalid requests."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests


root = Path(__file__).resolve().parents[1]
url = os.environ.get("MODEL_ROOT_URL", "http://127.0.0.1:7860").rstrip("/")
model = joblib.load(root / "backend_files/superkart_model.joblib")
batch = pd.read_csv(root / "Batch_Data_SuperKart.csv")
payload = batch.iloc[0].to_dict()
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)
    print("PASS:", name)


health = requests.get(url + "/health", timeout=30)
check("model health", health.status_code == 200 and health.json()["status"] == "ok")
single = requests.post(url + "/v1/predict", json=payload, timeout=30)
check("single request", single.status_code == 200)
np.testing.assert_allclose(single.json()["Sales"], model.predict(pd.DataFrame([payload]))[0], rtol=1e-6)
check("single prediction matches saved pipeline", True)

response = requests.post(
    url + "/v1/predictbatch",
    files={"file": ("batch.csv", batch.to_csv(index=False).encode(), "text/csv")},
    timeout=30,
)
check("batch request", response.status_code == 200)
check("all ten batch rows returned", set(response.json()) == {str(i) for i in range(len(batch))})
actual = np.array([response.json()[str(i)] for i in range(len(batch))])
np.testing.assert_allclose(actual, model.predict(batch), rtol=1e-6, atol=0.011)
check("batch predictions preserve row order and match saved pipeline", True)
check("unseen batch category reported", "Supermarket Type3" in response.headers.get("X-SuperKart-Warnings", ""))

bad_requests = {
    "missing column": {"Product_MRP": 100},
    "negative weight": {**payload, "Product_Weight": -1},
    "invalid area": {**payload, "Product_Allocated_Area": 2},
    "invalid number": {**payload, "Product_MRP": "not a number"},
    "empty category": {**payload, "Store_Size": ""},
    "fractional store age": {**payload, "Store_Age_Years": 1.5},
    "wrong JSON shape": [payload],
}
for label, data in bad_requests.items():
    invalid = requests.post(url + "/v1/predict", json=data, timeout=30)
    check(label + " rejected", invalid.status_code == 400 and "error" in invalid.json())

empty_upload = requests.post(url + "/v1/predictbatch", timeout=30)
check("missing upload rejected", empty_upload.status_code == 400)
empty_csv = requests.post(
    url + "/v1/predictbatch",
    files={"file": batch.iloc[:0].to_csv(index=False).encode()}, timeout=30,
)
check("empty CSV rejected", empty_csv.status_code == 400)

unknown = requests.post(
    url + "/v1/predict", json={**payload, "Store_Type": "Supermarket Type3"}, timeout=30
)
check("unknown category is visible", unknown.status_code == 200 and bool(unknown.json().get("warnings")))

evidence = {"backend_url": url, "passed": len(checks), "checks": checks}
(root / "results").mkdir(exist_ok=True)
(root / "results/deployment_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(f"Completed {len(checks)} checks against {url}.")
