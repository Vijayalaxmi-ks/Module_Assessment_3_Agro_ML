# Problem Framing Note: Crop Intervention Planning (Case A04)

## 1. Stakeholders & Business Context
* **Primary Stakeholder:** Precision Farming Operations & Agronomy Advisors.
* **Operational Goal:** Timely detection of crop stress/pest risks to schedule targeted interventions (chemical treatments, fertilization, or irrigation) before yield loss occurs.

## 2. Core Operational Decision
* **Decision:** Should a specific plot/field segment receive an intervention during the upcoming weekly cycle?
* **Unit of Analysis:** Field-Plot-Week (A specific plot observed over a 7-day interval).
* **Target Variable:** `intervention_required` (Binary: 0 = No Action, 1 = Targeted Intervention Required).

## 3. Operational Constraints & Asymmetric Harm Analysis
* **False Positive (FP - Unnecessary Treatment):** Financial cost of chemicals/labor, potential soil degradation, and environmental runoff.
* **False Negative (FN - Missed Outbreak):** Severe crop damage, yield reduction, and potential pest spreading to adjacent fields.
* **Cost Asymmetry:** A Missed Outbreak (FN) is typically 4× to 6× more costly than an Unnecessary Action (FP). Our ML evaluation metric must prioritize high **Recall** and **Cost-Weighted Loss**.