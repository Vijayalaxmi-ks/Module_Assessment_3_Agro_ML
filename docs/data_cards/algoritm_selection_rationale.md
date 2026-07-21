# Algorithm Selection Rationale & ML Architecture

## 1. Problem Statement & ML Task Formulation
* **Domain Context:** Case A04 — Precision Agriculture & Crop Intervention Planning.
* **Operational Goal:** Predict whether an agricultural plot requires a targeted management intervention (e.g., chemical treatment, nutrient application, irrigation) within a 7-day window to prevent crop yield loss.
* **Machine Learning Task Type:** **Supervised Binary Classification**
  * **Input Features ($X$):** Continuous micro-climate telemetry (temperature, humidity, soil moisture, soil pH, NPK levels, NDVI vegetation index, Vapor Pressure Deficit) and categorical plot descriptors (crop type, growth stage, geographic region).
  * **Target Label ($y$):** `intervention_required` $\in \{0: \text{No Action}, 1: \text{Intervention Required}\}$.

---

## 2. Theoretical Framework & Algorithm Selection Rationale

To ensure architectural rigor, four distinct algorithms were selected across linear, ensemble, and gradient-boosting families.

                     ┌─────────────────────────────────────────┐
                     │   Supervised Binary Classification      │
                     └────────────────────┬────────────────────┘
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       │                                                                     │
┌──────┴───────────────────────┐                         ┌───────────────────┴───────────────────┐
│     Parametric Baseline      │                         │     Non-Parametric Ensembles          │
└──────────────┬───────────────┘                         └───────────────────┬───────────────────┘
               │                                                             │
  ┌────────────┴────────────┐                  ┌─────────────────────────────┴─────────────────────────────┐
  │   Logistic Regression   │                  │                                                           │
  └─────────────────────────┘      ┌───────────┴───────────┐                                 ┌─────────────┴─────────────┐
                                   │ Bagging (Parallel)    │                                 │ Boosting (Sequential)     │
                                   └───────────┬───────────┘                                 └─────────────┬─────────────┘
                                               │                                                           │
                                  ┌────────────┴───────────┐                   ┌───────────────────────────┴───────────────────────────┐
                                  │     Random Forest      │                   │                                                       │
                                  └────────────────────────┘      ┌──────────┴───────────┐                              ┌──────────┴──────────┐
                                                                  │       XGBoost        │                              │        LightGBM     │
                                                                  └──────────────────────┘                              └─────────────────────┘


---

### **Algorithm 1: Logistic Regression**
* **ML Class:** Parametric Supervised Learning (Linear Model)
* **Mathematical Mechanism:** Models the log-odds of the positive class as a linear combination of input features using the Sigmoid (logistic) function:
  $$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{\beta}^T \mathbf{X})}}$$
* **Why Included:** Serves as the fundamental statistical **baseline**. It tests whether crop stress is linearly separable from raw telemetry indicators before adopting complex non-linear models.

---

### **Algorithm 2: Random Forest Classifier**
* **ML Class:** Non-Parametric Supervised Learning (Bagging Ensemble of Decision Trees)
* **Mathematical Mechanism:** Construct an ensemble of independent decision trees trained on bootstrap samples of the data ($BAGging$) with random feature sub-selection ($Feature\ Sampling$). Final prediction is obtained via majority voting:
  $$\hat{y} = \text{mode}\{f_1(X), f_2(X), \dots, f_B(X)\}$$
* **Why Included:** Highly resilient to overfitting, handles correlated agricultural features (e.g., temperature and humidity), and natively handles non-linear interactions without requiring strict feature scaling.

---

### **Algorithm 3: XGBoost (eXtreme Gradient Boosting)**
* **ML Class:** Non-Parametric Supervised Learning (Gradient Boosted Decision Trees)
* **Mathematical Mechanism:** Sequentially fits decision trees to minimize a second-order Taylor expansion of a regularized objective function:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
  Where $\Omega(f)$ penalizes model complexity ($\gamma T + \frac{1}{2}\lambda \sum w_j^2$).
* **Why Included:** Industry gold standard for tabular data. Its explicit $L_1/L_2$ regularization handles feature noise, while missing value awareness ensures stability under sensor dropouts.

---

### **Algorithm 4: LightGBM (Light Gradient Boosting Machine)**
* **ML Class:** Non-Parametric Supervised Learning (Histogram-based Gradient Boosting)
* **Mathematical Mechanism:** Uses **Leaf-wise (Best-First)** tree growth combined with **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)** to drastically accelerate training on tabular datasets.
* **Why Included:** Offers state-of-the-art execution speed and memory efficiency, making it ideal for real-time edge or serverless deployment in precision agricultural apps.

---

## 3. Comparative Evaluation Strategy

Given the **asymmetric economic loss** in crop management:
* **False Negative (FN):** Unnoticed pest or disease outbreak leading to severe yield loss ($\text{Cost} \approx 5\times$).
* **False Positive (FP):** Unnecessary chemical application ($\text{Cost} \approx 1\times$).

Models are evaluated on **Recall ($\frac{TP}{TP+FN}$)** and **F1-Score ($\frac{2 \cdot P \cdot R}{P + R}$)** to pick the Champion Model.