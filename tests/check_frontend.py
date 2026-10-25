# Use the actual frontend for one prediction, a CSV upload, and a download

# Read settings and save browser results
# import io  # DEPRECATED: CSV files are read directly from their downloaded paths
import json
import os
from pathlib import Path

# Compare the downloaded table with the original inputs and API values
import pandas as pd
import requests
from playwright.sync_api import sync_playwright

# Select the page and backend addresses for this run
root = Path(__file__).resolve().parents[1]
url = os.environ.get("FRONTEND_URL", "http://127.0.0.1:8501").rstrip("/")
api_url = os.environ.get("MODEL_ROOT_URL", "http://127.0.0.1:7860").rstrip("/")

# Reuse an installed browser, or let Playwright use its own Chromium installation
chrome = os.environ.get("CHROME_EXECUTABLE")
mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
if chrome is None and mac_chrome.exists():
    chrome = str(mac_chrome)

# Keep screenshots and downloaded predictions together
results = root / "results"
results.mkdir(exist_ok=True)

# Open the real page in a clean browser session
with sync_playwright() as browser_tools:
    browser = browser_tools.chromium.launch(headless=True, executable_path=chrome)
    page = browser.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(url, wait_until="networkidle")

    # GitHub asks first-time visitors to confirm the forwarded development port
    if page.title() == "Codespaces Access Port":
        page.get_by_text("Continue", exact=True).click()
        page.wait_for_load_state("networkidle")

    # Submit the default single-product example through the page
    page.get_by_role("heading", name="SuperKart System", exact=True).wait_for()
    page.get_by_role("button", name="Predict", exact=True).click()
    success = page.get_by_text("Predicted Product Store Sales Total:", exact=False)
    success.wait_for(timeout=60000)
    single_text = success.inner_text()

    # Check the displayed estimate against the API for the same default inputs
    example = pd.read_csv(root / "Batch_Data_SuperKart.csv").iloc[0].to_dict()
    single_api = requests.post(api_url + "/v1/predict", json=example, timeout=30)
    single_api.raise_for_status()
    assert f"{single_api.json()['Sales']:,.2f}" in single_text

    # Capture the visible single-product result
    success.scroll_into_view_if_needed()
    page.screenshot(path=str(results / "frontend_single.png"))

    # Submit the supplied file through the upload widget, not a direct API call
    page.locator('input[type="file"]').set_input_files(str(root / "Batch_Data_SuperKart.csv"))
    page.get_by_role("button", name="Predict for Batch", exact=True).click()
    page.get_by_text("Predictions completed for 10 products.", exact=True).wait_for(timeout=60000)
    page.get_by_text("Store_Type was not seen in training: Supermarket Type3", exact=True).wait_for()

    # The cloud page can show its download button before the table finishes painting
    page.get_by_test_id("stDataFrame").locator("canvas").first.wait_for(state="visible", timeout=60000)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Download predictions", exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(results / "frontend_batch.png"))

    # Download the CSV through the actual page control
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Download predictions", exact=True).click()
    download = download_info.value
    download.save_as(str(results / "SuperKart_UI_Download.csv"))

    # Compare every downloaded input row with the original upload
    downloaded = pd.read_csv(results / "SuperKart_UI_Download.csv")
    batch = pd.read_csv(root / "Batch_Data_SuperKart.csv")
    pd.testing.assert_frame_equal(downloaded.drop(columns="Predicted_Sales"), batch)

    # Compare the downloaded predictions with the backend's numbered results
    api = requests.post(
        api_url + "/v1/predictbatch",
        files={"file": batch.to_csv(index=False).encode()}, timeout=30,
    )
    api.raise_for_status()
    expected = [api.json()[str(i)] for i in range(len(batch))]
    assert downloaded.Predicted_Sales.tolist() == expected
    browser.close()

# Save the checked page address and the observed outcomes
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
