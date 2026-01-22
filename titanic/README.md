## 🚀 Current Status: High-Performance Pipeline
After experimenting with multiple algorithms, the **XGBoost Pipeline** achieved the highest accuracy.

### 📊 Model Performance Comparison (5-fold Cross-Validation)
1. **Baseline:** Decision Tree / Random Forest (~70-72%)
2. **Optimized Pipeline:** Random Forest Classifier (~82%)
3. **Optimized Pipeline (highest):** XGBoost Classifier (**Accuracy: 83.5%**)

### 💡 Technical Highlights
- **Automated Workflow:** Used `sklearn.pipeline.Pipeline` to handle imputation and modeling in one go.
- **Advanced Feature Engineering:** - Derived `FamilySize` and `FarePerPerson` to capture socio-economic dynamics.
  - Engineered `IsChild` and `Has_cabin` to capture survival priority.
- **Hyperparameter Tuning:** Optimized `n_estimators`, `learning_rate`, and `max_depth` for XGBoost to prevent overfitting.
