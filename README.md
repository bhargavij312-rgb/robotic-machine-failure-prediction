# Robotic / CNC Machine Failure Prediction (Predictive Maintenance)
**Learn Depth Academy LLP - Track 1 Capstone - Problem 21**
**Author:** Pooja Sharma

Predicts whether an automated machine (robot arm / CNC cell) will fail, from sensor and process readings.

## Dataset
AI4I 2020 Predictive Maintenance Dataset - S. Matzka, UCI ML Repository, 2020, https://doi.org/10.24432/C5HS5C (CC BY 4.0). 10,000 rows, 339 failures (3.39%), no missing values or duplicates. The dataset is synthetic, so results show feasibility, not real-world performance.

## Method
- Leakage control: failure-mode columns (TWF, HDF, PWF, OSF, RNF) excluded from features
- Engineered features: temperature difference, mechanical power, wear x torque
- Pipeline: median imputation, scaling, one-hot encoding
- Models compared with 5-fold stratified CV: Logistic Regression, KNN, Decision Tree
- Final model: Decision Tree (depth 5, balanced class weights)

## Results
| Model (CV) | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| LogReg (raw) | 0.804 | 0.137 | 0.235 | 0.897 |
| LogReg (engineered) | 0.841 | 0.162 | 0.271 | 0.923 |
| KNN | 0.273 | 0.786 | 0.405 | 0.881 |
| Decision Tree | 0.923 | 0.410 | 0.568 | 0.940 |

Test set: Recall 0.853, Precision 0.369, F1 0.516, ROC-AUC 0.883 (68 test failures).

## Run
pip install -r requirements.txt
python train.py
streamlit run app.py

## Links
GitHub: <YOUR_LINK> | LinkedIn: <YOUR_LINK>
