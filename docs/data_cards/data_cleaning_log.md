# Data Cleaning & Quality Assurance Log

**Total Records Processed:** 10000

## Actions Taken:
- Standardized casing for 'crop_type' and 'growth_stage'.
- Flagged 25 invalid temperature glitches (-999 C) as NaN.
- Flagged 38 invalid humidity glitches (>100%) as NaN.
- Imputed 25 missing/glitched values in 'avg_temp_c' using group medians.
- Imputed 544 missing/glitched values in 'avg_humidity_pct' using group medians.
- Imputed 495 missing/glitched values in 'soil_moisture_pct' using group medians.
- Imputed 494 missing/glitched values in 'soil_ph' using group medians.
- Imputed 538 missing/glitched values in 'ndvi_index' using group medians.

## Cleaned Data Schema Summary:
```text
<class 'pandas.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   field_id               10000 non-null  str    
 1   region                 10000 non-null  str    
 2   crop_type              10000 non-null  str    
 3   growth_stage           10000 non-null  str    
 4   avg_temp_c             10000 non-null  float64
 5   avg_humidity_pct       10000 non-null  float64
 6   soil_moisture_pct      10000 non-null  float64
 7   soil_ph                10000 non-null  float64
 8   nitrogen_ppm           10000 non-null  float64
 9   phosphorus_ppm         10000 non-null  float64
 10  potassium_ppm          10000 non-null  float64
 11  ndvi_index             10000 non-null  float64
 12  pest_pressure_score    10000 non-null  float64
 13  intervention_required  10000 non-null  int64  
dtypes: float64(9), int64(1), str(4)
memory usage: 1.1 MB

```
