# Model Comparison & Performance Evaluation

## Algorithmic Comparison Table

| Model                          |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:-------------------------------|-----------:|------------:|---------:|-----------:|----------:|
| Random Forest                  |     0.8815 |    0.577465 | 0.811881 |   0.674897 |  0.942778 |
| XGBoost                        |     0.901  |    0.69962  | 0.607261 |   0.650177 |  0.944978 |
| LightGBM                       |     0.896  |    0.672727 | 0.610561 |   0.640138 |  0.943025 |
| Logistic Regression (Baseline) |     0.8875 |    0.701031 | 0.448845 |   0.547284 |  0.880231 |

## Champion Model Selection
- **Selected Champion Model:** `Random Forest`
- **F1-Score:** `0.6749`
- **Primary Evaluation Justification:** In agricultural decision support, Recall is prioritized to minimize False Negatives (unnoticed crop risk).
