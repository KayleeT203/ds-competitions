# 🚢 Titanic Survival Prediction: Validated Ensemble Pipeline

This repository presents a structured approach to the Titanic survival prediction task, progressing from basic decision trees to a **validated ensemble solution**.

## 🚀 Project Status: Complete and Stable
The current iteration prioritises robustness and generalisability, utilising automated pipelines and weighted model blending.

### 📊 Model Performance Comparison
*Results based on 5-fold cross-validation (CV) to ensure reliability.*

| Model Stage | Algorithm | CV Accuracy |
| :--- | :--- | :--- |
| **Baseline** | Decision Tree / Random Forest | ~70–80% |
| **Optimised** | Random Forest Pipeline | ~81% |
| **Optimised (High)** | XGBoost Pipeline | **82–83%** |
| **Final Ensemble** | **XGBoost + Random Forest Blend** | **Most Stable** |

> **🏆 Kaggle Result:** The ensemble model achieves a leaderboard score of **~0.79–0.80 (Top ~7%)**, demonstrating strong alignment between local validation and unseen test data.

---

### 💡 Technical Highlights

#### 🛠️ Automated Pipelines
- Used `sklearn.pipeline.Pipeline` with `SimpleImputer` for reproducible preprocessing.
- Prevents **data leakage** and enables rapid experimentation with different models.

#### 🧠 Advanced Feature Engineering
Features were designed to capture socio-economic context and behavioural priorities:
- **Socio-economic indicators:** `FamilySize`, `FarePerPerson`
- **Survival priority signals:** `IsChild`, `HasCabin`, passenger titles extracted from names
- **Group dynamics:** shared ticket analysis and family grouping

#### ⚙️ Model Optimisation and Ensembling
- **Hyperparameter tuning:** refined `n_estimators`, `learning_rate`, `max_depth`, and regularisation parameters for XGBoost.
- **Ensembling strategy:** applied weighted probability averaging between XGBoost and Random Forest to reduce variance and improve robustness.

#### 🧪 Validation Discipline
- **Cross-validation over leaderboard tuning:** model selection relied strictly on 5-fold CV rather than leaderboard feedback.
- **Controlled feature growth:** additional features were only retained when they provided measurable improvements.

---

### 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-learn, XGBoost, Pandas, NumPy
- **Environment:** Kag
