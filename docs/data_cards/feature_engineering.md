# Feature Engineering & Data Pipeline Architecture

## Engineered Domain Features
- **`vpd_kpa`**: Vapor Pressure Deficit derived from ambient temperature and relative humidity.
- **`total_npk_ppm`**: Aggregate soil macronutrient load (N + P + K).
- **`ph_deviation`**: Absolute deviation of soil pH from ideal neutral state (7.0).

## Dataset Dimensions
- Train Set: 8000 samples, 22 features
- Test Set: 2000 samples, 22 features
- Target Class Distribution (Positive/Intervention Needed): 15.13%
