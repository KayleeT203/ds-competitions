# 🦟 DengAI: Predicting Disease Spread (Time-Series Regression)

This project tackles the challenge of predicting dengue fever outbreaks in San Juan and Iquitos using environmental and climate data. The focus is on **interpretable feature engineering** and **time-sensitive validation**.

## 🚀 Project Status: Active & Optimising
- **Current Rank:** #1504 / 7339 (**Top 20.5%**)
- **Current Performance:** MAE 25.181 (Validated via Time-Series Split)

### 📊 Local Validation Results
*Evaluated using `TimeSeriesSplit` to respect temporal order.*

| Model Type | San Juan (SJ) MAE | Iquitos (IQ) MAE |
| :--- | :--- | :--- |
| **XGBoost Pipeline** | ~18.66 | ~5.04 |
| **Random Forest Pipeline (Conservative)** | ~18.56 | ~5.25 |
| **Final Weighted Blend** | **Optimised per City** | **Stable Performance** |

---

### 💡 Technical Highlights

#### 🕰️ Temporal Feature Engineering (The "Mosquito Risk" Logic)
The core of the strategy is translating biological domain knowledge into features:
- **Lagged Indicators:** Shifted weather data by 2-6 weeks to account for the mosquito life cycle and virus incubation periods.
- **Rolling Statistics:** Used 4-week and 7-week moving averages to capture cumulative environmental stress (humidity/temperature).
- **Seasonal Signals:** Engineered `wet_season` and `spike_window` features to help the model identify peak outbreak months.

#### 🏗️ City-Specific Architectures
Recognising that San Juan (coastal) and Iquitos (rainforest) have distinct climates, I implemented:
- **Dual Pipelines:** Completely separate preprocessing and model parameters for each city.
- **Strategic Scaling:** Used `MinMaxScaler` fitted on training data and applied to test data to prevent information leakage while normalising high-variance weather metrics.

#### 🧪 Time-Series Validation Discipline
- **Temporal Integrity:** Replaced standard K-Fold with `TimeSeriesSplit` to ensure the model never "trained on the future" to predict the past.
- **Error Analysis:** Focused on Mean Absolute Error (MAE) to maintain a linear understanding of prediction deviations in a public health context.

---

### 🛠️ Project Structure
- `main.py`: Main execution script including pipeline definition and ensembling.
- `utils.py`: Modularised functions for feature engineering and data cleaning.
