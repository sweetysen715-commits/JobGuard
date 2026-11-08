# JobGuard

JobGuard is a machine-learning-powered web application that helps job seekers identify potentially fraudulent job postings and analyze resume skill gaps.

## Features

### 1. Job Scam Detection
- Random Forest classification model
- Fraud probability estimation
- Tuned classification threshold
- SHAP-based model explainability

### 2. Resume Skill Gap Analysis
- PDF and DOCX resume parsing
- Skill extraction and normalization
- Exact and semantic skill matching
- Matched, missing, and extra skill identification
- Career readiness score
- Learning recommendations and suggested learning paths

### 3. Job Market Insights
- Skill frequency analysis using legitimate job postings
- Dataset-based market skill insights

## Technology Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- TF-IDF, Sentence Transformers
- SHAP
- Flask
- PyMuPDF, python-docx

## Model Evaluation

The selected Random Forest model achieved the following results on a held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.01% |
| Fraud Precision | 82% |
| Fraud Recall | 76% |
| Fraud F1-score | 79% |
| PR-AUC | 0.855 |
| ROC-AUC | 0.978 |

These results are dataset-specific. Predictions are estimates, not guarantees.

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd JobGuard
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

Open http://127.0.0.1:5000/ in your browser.

## Limitations

- Fraud predictions are not guarantees.
- Skill extraction depends on the skill dictionary and semantic matching model.
- Market insights reflect the analyzed dataset, not live job-market statistics.
- Career readiness is a job-specific skill-coverage score, not a hiring prediction.

## Author

Computer Science and Engineering Student  
Haldia Institute of Technology
=======
ML-powered job scam detection and resume skill gap analyzer.
>>>>>>> d075d2aa112fd9432b7e8d2f9ac571513a2d5614
