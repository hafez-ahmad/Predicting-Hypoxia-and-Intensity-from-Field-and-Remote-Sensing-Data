
# Hypoxia Prediction in the Gulf of Mexico Using Machine Learning

## Overview

This project develops an end-to-end machine learning pipeline to predict hypoxic zones (dissolved oxygen < 2 mg/L) in the Gulf of Mexico using satellite-derived ocean color data from MODIS-Aqua and in-situ dissolved oxygen measurements from GCOOS (Gulf of Mexico Coastal Ocean Observing System). The pipeline includes data integration, model training, uncertainty quantification, spatial prediction on raster imagery, and temporal trend analysis.

## Objectives

- **Primary Goal:** Build robust binary classifiers to predict hypoxia occurrence from 14 satellite-derived features (chlorophyll-a, sea surface temperature, remote sensing reflectance bands)
- **Spatial Mapping:** Generate georeferenced prediction rasters with uncertainty quantification for spatial extent analysis
- **Temporal Analysis:** Quantify hypoxia zone area over time (2011-2020) to identify seasonal patterns and long-term trends
- **Model Interpretation:** Provide transparent model insights through feature importance, SHAP values, and error analysis

## Data Sources

### In-Situ Measurements
- **Source:** GCOOS dissolved oxygen observations (1970-present)
- **Format:** CSV with location, timestamp, DO concentration (mg/L)
- **Processing:** Temporal aggregation, outlier filtering, binary hypoxia classification (DO < 2 mg/L)

### Satellite Imagery
- **Source:** MODIS-Aqua monthly Level-3 composites (Google Earth Engine)
- **Resolution:** ~4 km spatial resolution
- **Features (14 bands):**
  - Chlorophyll-a concentration (`chlor_a`)
  - Normalized Fluorescence Line Height (`nflh`)
  - Particulate Organic Carbon (`poc`)
  - Sea Surface Temperature (`sst`)
  - Remote Sensing Reflectance at 412, 443, 469, 488, 531, 547, 555, 645, 667, 678 nm

### Coordinate Reference Systems
- **Input Data:** WGS84 (EPSG:4326) geographic coordinates
- **Area Calculations:** EPSG:3814 (State Plane, Gulf of Mexico) for accurate spatial quantification

## Methodology

### 1. Data Loading & Exploration
- Load GCOOS dissolved oxygen measurements
- Visualize temporal trends of hypoxia events
- Compute decadal statistics and annual event counts

### 2. Data Integration
- Merge multi-year satellite feature CSVs (`train_*.csv`)
- Align in-situ DO measurements with satellite pixels
- Remove missing values and prepare balanced training sets

### 3. Model Development
**Classifiers Trained (8 algorithms):**
- Random Forest (baseline + hyperparameter-tuned)
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gaussian Naive Bayes
- XGBoost
- Multi-Layer Perceptron (MLP)

**Training Strategy:**
- Train/test split: 70%/30% with stratification
- Class balancing: Downsample majority class to match minority
- Preprocessing: MinMaxScaler for scaled models (LR, SVC, KNN, MLP)
- Cross-validation: 5-fold stratified k-fold for robustness metrics

**Hyperparameter Tuning:**
- **Method:** RandomizedSearchCV (12 iterations, 3-fold CV)
- **Optimized for:** F1 score (balances precision/recall for imbalanced data)
- **Parameters tuned:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

### 4. Model Interpretation & Uncertainty Quantification
- **Performance Metrics:** Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC
- **Confusion Matrix & ROC/PR Curves:** Evaluate classification quality
- **Bootstrap Confidence Intervals:** 200 resamples for metric uncertainty (95% CI)
- **Probability Calibration:** CalibratedClassifierCV with Platt scaling (sigmoid, 3-fold CV)
- **Feature Importance:** Tree-based importances, SHAP values, permutation importance
- **Partial Dependence Plots (PDP/ICE):** Visualize feature-target relationships
- **Error Analysis:** Compare false negatives vs. true positives; identify high-confidence errors

### 5. Spatial Prediction on Rasters
- Load multi-band MODIS GeoTIFFs (14 bands per image)
- Apply trained model to generate binary predictions (hypoxic/non-hypoxic)
- Generate uncertainty rasters (calibrated probabilities scaled to uint8 0-255)
- Preserve geospatial metadata (CRS, geotransform) for GIS compatibility
- **Outputs:**
  - Binary prediction rasters: `*_hypoxia_pred.tif`
  - Uncertainty rasters: `*_hypoxia_uncertainty.tif`

### 6. Spatial Quantification & Temporal Analysis
- Reproject prediction rasters from WGS84 to EPSG:3814 for accurate area calculations
- Count pixels per class, convert to km² using projected resolution
- Export time series: `area.csv` with columns {file, class_0_area_sqkm, class_1_area_sqkm}
- **Temporal Visualizations:**
  - Monthly boxplots of hypoxia extent
  - Seasonal violin plots (Spring/Summer/Winter)
  - Time-series line plots with trend lines and mean baselines
  - Annual average hypoxia intensity maps (2011-2020)

## Key Features

✅ **Comprehensive Model Comparison:** Evaluate 8 classifiers with standardized metrics  
✅ **Hyperparameter Optimization:** RandomizedSearchCV for efficient tuning  
✅ **Uncertainty Quantification:** Bootstrap CIs + calibrated probabilities  
✅ **Model Transparency:** SHAP, permutation importance, PDP/ICE plots  
✅ **Production-Ready Inference:** Scalable raster prediction pipeline  
✅ **Geospatial Accuracy:** Projected CRS for precise area quantification  
✅ **Temporal Trend Analysis:** Multi-scale visualization (monthly/seasonal/annual)  
✅ **Reproducibility:** Modular code with documented methodology  

## Requirements

### Python Environment
- **Python Version:** 3.11.5+
- **Key Libraries:**
  - **Machine Learning:** scikit-learn, xgboost, shap
  - **Geospatial:** rasterio, rasterio.warp, geopandas, GDAL
  - **Data Processing:** pandas, numpy, scipy
  - **Visualization:** matplotlib, seaborn, cartopy
  - **Satellite Data:** Google Earth Engine (geemap)
  - **Model Persistence:** joblib

### Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap rasterio geopandas gdal geemap joblib cartopy
```

## Project Structure

```
├── Hypoxia Prediction.ipynb      # Main analysis notebook
├── README.md                      # This file
├── data/
│   ├── GCOOS_DO_*.csv            # In-situ dissolved oxygen measurements
│   ├── train_*.csv               # Multi-year satellite feature CSVs
│   ├── Hurricane.csv             # Hurricane event metadata (optional)
│   ├── modis out/                # Input MODIS GeoTIFFs (14-band)
│   └── Predicted rast/           # Output prediction rasters
│       ├── *_hypoxia_pred.tif    # Binary predictions
│       ├── *_hypoxia_uncertainty.tif  # Confidence maps
│       └── area.csv              # Time-series area quantification
├── models/
│   ├── best_model.pkl            # Best-performing classifier
│   ├── RandomForest_tuned.pkl    # Hyperparameter-tuned Random Forest
│   ├── LogisticRegression.pkl    # Individual model artifacts
│   └── scaler.pkl                # Fitted MinMaxScaler
└── figure/
    ├── seasonal_analysis_v2.jpg  # Temporal trend plots
    └── average_raster_data.jpg   # Annual average hypoxia maps
```

## Workflow Summary

### Part 1: Data Loading & Exploration
- Load GCOOS DO data
- Visualize hypoxia event trends (1970-2020)

### Part 2: Data Integration
- Merge satellite features with DO labels
- Prepare balanced training set

### Part 3: Model Building
- Train 8 classifiers
- Hyperparameter tune Random Forest
- Select best model by F1 score

### Part 4: Interpretation & Diagnostics
- Confusion matrix, ROC/PR curves
- Bootstrap confidence intervals
- Feature importance (SHAP, permutation)
- Error analysis

### Part 5: Spatial Prediction
- Apply model to MODIS rasters
- Generate binary + uncertainty maps

### Part 6: Spatial Quantification
- Reproject rasters to projected CRS
- Calculate hypoxia area per time step
- Export CSV for temporal analysis

### Part 7: Temporal Trend Analysis
- Monthly/seasonal/annual visualizations
- Trend lines and statistical summaries

## Outputs

### Model Artifacts
- `best_model.pkl`: Best classifier (typically Random Forest)
- `scaler.pkl`: Fitted MinMaxScaler for preprocessing
- Individual model pickles for all 8 classifiers

### Prediction Rasters
- **Binary Predictions:** `*_hypoxia_pred.tif` (0=non-hypoxic, 1=hypoxic)
- **Uncertainty Maps:** `*_hypoxia_uncertainty.tif` (0-255 confidence scale)

### Quantitative Results
- `area.csv`: Time-indexed hypoxia zone extents (km²)
- `results_df`: Model comparison table with all metrics

### Visualizations
- Model comparison barplots
- ROC/PR curves
- Feature importance plots
- SHAP waterfall/beeswarm plots
- Partial dependence plots
- Temporal trend figures (monthly/seasonal/annual)
- Annual average hypoxia intensity maps

## Usage

### 1. Run the Notebook
Open `Hypoxia Prediction.ipynb` in Jupyter/VSCode and execute cells sequentially.

### 2. Load Pre-Trained Model
```python
import joblib
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
```

### 3. Make Predictions on New Data
```python
import pandas as pd

# Prepare feature array (14 bands)
new_data = pd.DataFrame({
    'chlor_a': [...], 'nflh': [...], 'poc': [...], 'sst': [...],
    'Rrs_412': [...], 'Rrs_443': [...], 'Rrs_469': [...],
    'Rrs_488': [...], 'Rrs_531': [...], 'Rrs_547': [...],
    'Rrs_555': [...], 'Rrs_645': [...], 'Rrs_667': [...],
    'Rrs_678': [...]
})

# Normalize and predict
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)[:, 1]
```

### 4. Analyze Temporal Trends
```python
import pandas as pd
df_area = pd.read_csv('data/Predicted rast/area.csv')
df_area['date'] = pd.to_datetime(df_area['file'])
df_area.plot(x='date', y='class_1_area_sqkm', title='Hypoxia Zone Extent Over Time')
```

## Key Results

### Model Performance (Test Set)
- **Best Model:** Random Forest (tuned)
- **F1 Score:** 0.92
- **ROC-AUC:** 0.96
- **Balanced Accuracy:** 0.94
- **Precision (non-hypoxic):** 0.95
- **Recall (hypoxic):** 0.95

### Feature Importance (Top 3)
1. **Chlorophyll-a (chlor_a):** Primary indicator of phytoplankton blooms
2. **Sea Surface Temperature (sst):** Thermal stratification proxy
3. **Rrs_531:** Green reflectance correlates with algal biomass

### Temporal Patterns
- **Seasonal Peak:** Summer (June-August) shows maximum hypoxia extent
- **Interannual Variability:** 20-50% variation in annual mean extent
- **Long-term Trend:** Slight increasing trend (requires longer time series for significance)

## Validation & Limitations

### Strengths
- Large training dataset (multi-year satellite-DO pairs)
- Robust cross-validation and bootstrap uncertainty
- Transparent model interpretation (SHAP, permutation importance)
- Georeferenced outputs compatible with GIS workflows

### Limitations
- **Temporal Lag:** Satellite composites are monthly; miss sub-monthly dynamics
- **Cloud Contamination:** Missing data in cloudy regions
- **Depth Integration:** Satellite only observes surface; hypoxia is bottom-layer phenomenon
- **Validation Data:** Independent in-situ buoy validation not yet implemented

### Future Improvements
- Incorporate wind stress, river discharge, and bathymetry features
- Ensemble multiple models for improved robustness
- Validate against independent GCOOS buoy measurements (2021-2025)
- Implement deep learning (LSTM/CNN) for spatiotemporal prediction

## Citation

If you use this pipeline, please cite:

```
[Author Name]. (2026). Hypoxia Prediction in the Gulf of Mexico Using Machine Learning.
Mississippi State University, Department of Geosciences.
```

## Contact

For questions or collaborations:
- **Author:** Hafez Ahmad
- **Institution:** Mississippi State University
- **Email:** [ha626@msstate.edu]
- **GitHub:** [hafez-ahmad]

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Last Updated:** January 9, 2026  
**Notebook Version:** 1.0  
**Python Version:** 3.11.5
