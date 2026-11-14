# ANN-Based House Price Regression — Full Project

## What is included
- `train_model.py`: preprocessing, ANN training, baseline comparison, metrics and charts.
- `app.py`: Streamlit web frontend with a house-price prediction form and performance dashboard.
- `data/train.csv`: place the Kaggle training CSV here.
- `artifacts/`: generated automatically after model training.

## Setup in VS Code
Open this folder in VS Code. Activate your virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the additional frontend dependency and project packages:

```bash
python -m pip install -r requirements.txt
```

Download Kaggle `train.csv` and put it at `data/train.csv`.

Train the model:

```bash
python train_model.py
```

Launch the website:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501).

## Important
The Kaggle dataset contains houses in Ames, Iowa, USA, and prices are in US dollars.
The form exposes selected features; all remaining columns use training-set median/mode defaults. Predictions are therefore educational estimates, not professional valuations.
Do not claim an improvement unless measured metrics improve. The script reports ANN and a simple median baseline on the same held-out test split.
