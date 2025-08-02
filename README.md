# 📘 GHG Emissions Analysis Project 

## 🔰 Project Title:
**Analyzing Greenhouse Gas (GHG) Emissions Across US Industries**

---

## 📌 Objective:
The objective of this project is to analyze and predict greenhouse gas emissions across various industries in the United States using data science and machine learning. The primary goals include:

- Cleaning and merging raw datasets from multiple sources.
- Identifying key polluting industries.
- Understanding time-based emission trends.
- Predicting future emissions using machine learning.
- Visualizing insights and model performance.

---

## 🧠 Project Summary:

This end-to-end data project explores greenhouse gas emissions data from US industries and applies machine learning for predictive analysis. It includes:

- Data pre-processing and merging from multiple raw files.
- Feature engineering and encoding of categorical data.
- Visualization of top emitters and correlation analysis.
- Training of an XGBoost Regressor model.
- Evaluation using R² score and Mean Squared Error.
- Saving final outputs and the trained model locally.

---

## 📊 Key Outputs:

| Filename                      | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `top5_industries_emissions.png` | Line chart for emissions by top 5 industries                              |
| `correlation_heatmap.png`    | Correlation matrix of key numeric features                                 |
| `feature_importance.png`     | Feature importance plot from XGBoost model                                 |
| `predicted_vs_actual.png`    | Comparison of predicted vs actual emissions                                |
| `final_model.pkl`            | Serialized trained XGBoost regression model                                |

---

## 📈 Results Summary:

- ✅ **Cleaned Records:** 1848  
- ✅ **Model Used:** XGBoost Regressor  
- ✅ **R² Score:** 0.9966  
- ✅ **MSE:** 0.0002  
- ✅ **Top Industries Identified:** Based on cumulative emission contributions  
- ✅ **Visual Insights:** Time-series trends, feature correlations, and prediction comparisons

---

## ⚙️ Tools & Technologies:

- **Python**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **VS Code**

---

## ✅ Conclusion:

This project demonstrates a complete workflow for analyzing greenhouse gas emissions using Python. It highlights how data science tools can help uncover pollution patterns and forecast future emissions. The approach can support environmental policy, academic research, and industry monitoring.

---