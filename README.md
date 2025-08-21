# Book-to-Market Equity Strategy

This project was part of my ACFI234 Theory of Finance II module.  
It investigates the Book-to-Market (B/M) ratio anomaly and its ability to predict stock returns using different portfolio and regression approaches.

## Contents
- `TheoryofFinance.py` → Python code for analysis
- `ff_data.sqlite` → SQLite dataset (stock and factor data)
- `Theory of Finance II - report.docx` → Full report with methodology, results, and discussion
- `data_assignment.sqlite` → Full dataset 

## Methods
- Fama-MacBeth two-step regressions
- Equally and value-weighted quintile portfolio analysis
- Long-short portfolio construction
- Factor attribution with Fama-French 3- and 5-factor models
- Dependent double-sorting (B/M & investment)

## Key Findings
- B/M ratio is a statistically significant predictor of future stock returns (t=3.78, p=0.002).
- Equally weighted long-short strategy generated strong abnormal returns (~0.81% monthly).
- Value-weighted portfolios reduced profitability → small-cap stocks are key drivers of the anomaly.
- Fama-French models explain some returns, but alpha remains significant.
- Dependent double sorting confirms robustness, but investment conditioning reduces profitability.

## Requirements
- Python 3.x
- pandas, numpy, statsmodels, sqlite3, matplotlib

## How to Run
```bash
python TheoryofFinance.py
