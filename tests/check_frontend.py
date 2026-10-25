"""Use the actual frontend for one prediction, a CSV upload, and a download."""

import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
from playwright.sync_api import sync_playwright


root = Path(__file__).resolve().parents[1]
url = os.environ.get("FRONTEND_URL", "http://127.0.0.1:8501").rstrip("/")
api_url = os.environ.get("MODEL_ROOT_URL", "http://127.0.0.1:7860").rstrip("/")
chrome = os.environ.get("CHROME_EXECUTABLE")
mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
if chrome is None and mac_chrome.exists():
    chrome = str(mac_chrome)
results = root / "results"
results.mkdir(exist_ok=True)

with sync_playwright() as browser_tools:
    browser = browser_tools.chromium.launch(headless=True, executable_path=chrome)
    page = browser.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(url, wait_until="networkidle")
    # GitHub asks first-time visitors to confirm the forwarded development port
    if page.title() == "Codespaces Access Port":
        page.get_by_text("Continue", exact=True).click()
        page.wait_for_load_state("networkidle")
    page.get_by_role("heading", name="SuperKart System", exact=True).wait_for()
    page.get_by_role("button", name="Predict", exact=True).click()
    success = page.get_by_text("Predicted Product Store Sales Total:", exact=False)
    success.wait_for(timeout=60000)
    single_text = success.inner_text()
    success.scroll_into_view_if_needed()
    page.screenshot(path=str(results / "frontend_single.png"))

    # Submit the supplied file through the upload widget, not a direct API call
    page.locator('input[type="file"]').set_input_files(str(root / "Batch_Data_SuperKart.csv"))
    page.get_by_role("button", name="Predict for Batch", exact=True).click()
    page.get_by_text("Predictions completed for 10 products.", exact=True).wait_for(timeout=60000)
    page.get_by_text("Store_Type was not seen in training: Supermarket Type3", exact=True).wait_for()
    page.get_by_role("button", name="Download predictions", exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(results / "frontend_batch.png"))
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Download predictions", exact=True).click()
    download = download_info.value
    download.save_as(str(results / "SuperKart_UI_Download.csv"))
    downloaded = pd.read_csv(results / "SuperKart_UI_Download.csv")
    batch = pd.read_csv(root / "Batch_Data_SuperKart.csv")
    pd.testing.assert_frame_equal(downloaded.drop(columns="Predicted_Sales"), batch)
    api = requests.post(
        api_url + "/v1/predictbatch",
        files={"file": batch.to_csv(index=False).encode()}, timeout=30,
    )
    api.raise_for_status()
    expected = [api.json()[str(i)] for i in range(len(batch))]
    assert downloaded.Predicted_Sales.tolist() == expected
    browser.close()

evidence = {
    "frontend_url": url,
    "single_result": single_text,
    "batch_rows": len(downloaded),
    "unseen_category_warning": True,
    "download_matches_api": True,
    "original_row_order_preserved": True,
}
(results / "frontend_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence, indent=2))
