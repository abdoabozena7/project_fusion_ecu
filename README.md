# CTR Prediction with Data Fusion and Ensemble Learning

This repository demonstrates a complete machine‑learning pipeline for click‑through rate (CTR) prediction using multiple data sources.  The goal is to build a robust model that can estimate the probability that a user clicks on an advertisement based on their demographic profile, historic browsing behaviour and ad attributes.

The project follows a structured workflow:

1. **Data ingestion** – Raw CSV files from [Alibaba Taobao Advertising Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408) are placed into the `data/raw` directory.  They include user demographics (`user_profile.csv`), ad metadata (`ad_feature.csv`), click logs (`raw_sample.csv`) and behavioural logs (`behavior_log.csv`).  *You must download these files yourself from Kaggle or TianChi and copy them into `data/raw`.*

2. **Data cleaning** – Missing values are imputed, duplicate rows dropped and suspicious outliers removed.  Timestamp columns are converted into datetime objects.  Each step is documented in the notebook `02_data_cleaning.ipynb`.

3. **Exploratory data analysis (EDA) and fusion** – We explore distributions of individual variables, examine relationships between variables and build preliminary insights.  Multiple data sources are fused using three strategies:

   * **Early Fusion:** merge user, ad and click logs on common keys (e.g. `user id` and `adgroup id`), producing a single, enriched table.
   * **Late Fusion:** train separate models on each data source and combine their predictions via stacking or voting.
   * **Hybrid Fusion:** derive statistical features from separate sources (e.g. average click rate per ad, browsing frequency per user) and append them to the main table before training.

   All visualisations and intermediate data sets are stored in `03_eda_fusion.ipynb`.  We generate histograms, box plots, correlation heatmaps and feature importance charts.

4. **Preprocessing and feature engineering** – Categorical variables are encoded using one‑hot encoding or frequency encoding.  Numeric variables are scaled.  We engineer additional features such as polynomial combinations and statistical aggregates.  The preprocessed data is split into training and test sets with stratification on the target label.

5. **Model development** – Several baseline classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost and CatBoost) are trained.  We then build ensemble models including bagging, boosting, voting and stacking classifiers.  Hyper‑parameters are tuned via cross‑validation.  The best models are saved to `models/` for later use.

6. **Evaluation** – The models are compared using metrics appropriate for classification: Area Under the ROC Curve (AUC), F1‑score, accuracy, precision, recall and confusion matrices.  We also perform error analysis to understand where the models succeed or fail.  Results are summarised in `06_model_evaluation.ipynb`.

7. **Deployment** – A Streamlit application is provided in `07_deployment_streamlit.ipynb` to demonstrate how to load the trained model, accept user input and display a predicted click probability.  This notebook can be exported to a standalone script for deployment.

## Project structure

```
project_fusion_ensemble/
├── data/
│   ├── raw/           # Place downloaded CSV files here (user_profile.csv, ad_feature.csv, raw_sample.csv, behavior_log.csv)
│   └── processed/     # Intermediate and cleaned data sets (generated programmatically)
├── notebooks/
│   ├── 01_data_overview.ipynb        # Initial exploration and summary of each data source
│   ├── 02_data_cleaning.ipynb        # Handling missing data, duplicates and outliers
│   ├── 03_eda_fusion.ipynb           # Exploratory data analysis and data fusion strategies
│   ├── 04_preprocessing.ipynb        # Encoding, scaling, feature engineering and handling imbalance
│   ├── 05_model_training.ipynb       # Training baseline and ensemble models
│   ├── 06_model_evaluation.ipynb     # Evaluating and comparing model performance
│   └── 07_deployment_streamlit.ipynb # Streamlit app for inference
├── models/           # Persisted trained models (saved during execution of notebooks)
├── requirements.txt  # Python package requirements
└── README.md         # This document
```

## Getting started

1. **Install dependencies:** create a virtual environment and install the packages listed in `requirements.txt`.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download data:** obtain the four CSV files from Kaggle or TianChi.  Copy them into `project_fusion_ensemble/data/raw/` with the same names used in the notebooks.  Make sure not to commit the raw data to version control.

3. **Run the notebooks:** open each notebook in order.  Execute all cells to reproduce the results.  The notebooks are heavily commented to explain each step, from data inspection through feature engineering and model building, to evaluation and deployment.

4. **Deploy the model:** after training and evaluation, the best model is saved in `models/`.  The Streamlit notebook demonstrates how to load this model and create a simple web interface for inference.  You can export the notebook to a Python script using Jupyter’s `File → Download as → Python` or convert using `nbconvert`.

## Notes

* This project avoids deep learning and image models.  The focus is on classical machine‑learning algorithms and statistical techniques suitable for tabular data.
* The notebooks are designed to be modular.  You can add or skip sections depending on your coursework requirements.
* Visualisation is performed with Matplotlib, Seaborn and Plotly.  Feature importances are extracted from tree‑based models and displayed as bar charts.
* The code is written in a notebook format for ease of explanation.  For production use, consider refactoring into reusable Python modules and scripts.

Feel free to explore, modify and extend this project.  Good luck with your studies!