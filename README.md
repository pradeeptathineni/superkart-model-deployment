# SuperKart model deployment

Estimate revenue for a product-store record using the supplied SuperKart data.
The notebook covers data inspection, revenue patterns, feature preparation,
Random Forest and XGBoost evaluation, and hyperparameter tuning. The selected
pipeline is saved and served through Flask and Streamlit in separate Docker
containers.

The dataset has no dates, quarter labels, or stated currency. Predictions use
the original revenue units and observation period. A next-quarter forecast
requires dated sales data and a time-based evaluation.

## Run the application

```sh
docker compose up --build -d --wait
docker compose ps
```

Open http://localhost:8501 for the form and CSV upload. The API listens on
http://localhost:7860. The frontend calls the backend over the Docker network.

The saved model and its metadata are included in the private deployment
repository. No API key is required to run inference.

## GitHub Codespaces

[Open the deployed app](https://superkart-deployment-pj6wp64jxg727xpw-8501.app.github.dev)
or [open the Codespace](https://superkart-deployment-pj6wp64jxg727xpw.github.dev).
The API is at https://superkart-deployment-pj6wp64jxg727xpw-7860.app.github.dev.
GitHub may show a Continue page on the first app visit. The Codespace is set to
stop after 30 minutes of inactivity; reopen it before using these URLs again.

Create a Codespace from this repository. The dev-container configuration starts
the two services with Docker Compose. Open forwarded port 8501 for the app.
Set port 7860 to Public in the Ports panel when an external notebook needs to
call the API. The forwarded addresses work only while the Codespace is running.

The dev-container image removes an unused Yarn package source whose missing
signing key prevented Docker setup. Package signature verification remains
enabled; this Python project does not use Yarn.
The SSH feature allows GitHub CLI to inspect container status during deployment.

## Run the notebook

Both supplied CSV files belong beside the notebook and are included in the private
repository. SuperKart.csv is used for training, and Batch_Data_SuperKart.csv
contains the ten supplied inference examples.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name superkart --display-name 'Python 3'
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/jupyter nbconvert --to notebook --execute --inplace Full_Code_SuperKart_Model_Deployment_Notebook_solution.ipynb --ExecutePreprocessor.kernel_name=superkart --ExecutePreprocessor.timeout=1800
.venv/bin/jupyter nbconvert --to html Full_Code_SuperKart_Model_Deployment_Notebook_solution.ipynb
```

On macOS, XGBoost also requires an available libomp runtime; a normal Homebrew
installation uses `brew install libomp`. The Linux Docker image installs
libgomp1 itself. Docker must be running for the notebook's deployment cells.

For Colab, upload the notebook and CSV files and run from their folder. Restart
the runtime after installing the dependencies. Set MODEL_ROOT_URL to the running
Codespace's port-7860 URL and FRONTEND_URL to its port-8501 URL before inference.
The notebook records whether it actually used local Docker or a forwarded URL.

## Inputs and outputs

The ten input columns are Product_Weight, Product_Sugar_Content,
Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type,
Store_Type, Product_Id_char, Store_Age_Years, and Product_Type_Category.
Store age uses a fixed 2025 reference year for both training and prediction.

- GET /health checks whether the API loaded its model.
- POST /v1/predict accepts one JSON object and returns a numeric Sales value.
- POST /v1/predictbatch accepts a multipart CSV in the file field and returns
  a row-indexed prediction dictionary.

The API rejects missing columns, invalid numeric inputs, and empty categories.
Unknown categories are processed by the encoder and reported for review.
The batch response carries category warnings in its X-SuperKart-Warnings
header, keeping warnings separate from the numbered predictions.

The supplied batch contains Supermarket Type3, which was not present during
training. It also combines store attributes in ways not represented by the four
training stores. Its predictions demonstrate the request path, not validated
accuracy for those new stores. No observed sales are provided for that batch.

## Verification

```sh
.venv/bin/python tests/check_deployment.py
```

Set MODEL_ROOT_URL to check a forwarded Codespace endpoint instead. The check
compares live single and batch predictions with the actual saved pipeline,
verifies row order, checks the unknown-category warning, and exercises invalid
requests. The notebook also verifies that saving and reloading the pipeline
leaves predictions unchanged. Model comparison tables and batch predictions
are saved in results/.

For the optional browser check, install Playwright and its Chromium browser,
then run tests/check_frontend.py. On this Mac the check uses the installed
Google Chrome; CHROME_EXECUTABLE can select another compatible browser.

```sh
.venv/bin/python -m pip install playwright==1.64.1
.venv/bin/python -m playwright install chromium
.venv/bin/python tests/check_frontend.py
```

The notebook records single and batch predictions, model metrics, and browser
checks. Inactive lines marked CHANGED or DEPRECATED explain why an alternative
was kept or replaced.
