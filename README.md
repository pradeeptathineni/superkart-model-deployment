# SuperKart model deployment

Estimate revenue for a product-store record using the supplied SuperKart data.
The completed notebook follows the Great Learning / UT Austin McCombs course
workflow: inspect the data, explore revenue patterns, prepare features, compare
and tune Random Forest and XGBoost, save the selected pipeline, and serve it
through Flask and Streamlit in separate Docker containers.

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

Create a Codespace from this repository. The dev-container configuration starts
the two services with Docker Compose. Open forwarded port 8501 for the app.
Set port 7860 to Public in the Ports panel when an external notebook needs to
call the API. The forwarded addresses work only while the Codespace is running.

The dev-container image removes an unused Yarn package source whose missing
signing key prevented Docker setup. Package signature verification remains
enabled; this Python project does not use Yarn.
The SSH feature allows GitHub CLI to inspect container status during deployment.

## Run the notebook

The original SuperKart.csv belongs beside the notebook; it remains local rather
than being republished in this deployment repository. Batch_Data_SuperKart.csv
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
Store age uses the fixed 2025 reference year from the course notebook.

- GET /health checks whether the API loaded its model.
- POST /v1/predict accepts one JSON object and returns a numeric Sales value.
- POST /v1/predictbatch accepts a multipart CSV in the file field and returns
  the course's row-indexed prediction dictionary.

The API rejects missing columns, invalid numeric inputs, and empty categories.
Unknown categories are processed by the encoder and reported for review.
The batch response carries category warnings in its X-SuperKart-Warnings
header, preserving the course's prediction-response shape.

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

Course masters remain unchanged. Changes to supplied code are explained in
comments and in the plain bullet notes below the corresponding notebook cells.
Public comparison sources and exact revisions are recorded in solution-links.txt.
