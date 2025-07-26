# 📘 GHG Emissions Analysis Project  
## 🧭 Status: **Phase 2**

---

## 🔰 Project Title:
**Analyzing Greenhouse Gas (GHG) Emissions Across US Industries**

---

## 📌 Objective:
To analyze greenhouse gas emissions data and predict future trends by:

- Merging, cleaning, and standardizing raw emission factor datasets.
- Identifying key contributors and emission trends by industry.
- Training a machine learning model to predict total emissions.
- Visualizing actual vs. predicted emissions for performance evaluation.

---

## ✅ Phases Completed

### 🧪 **Phase 1: Data Cleaning & Initial Analysis**
- Imported raw emissions datasets and validated required columns.
- Cleaned data, handled missing values, standardized column names.
- Converted `Year` column to integer format.
- Identified **top 5 polluting industries** based on cumulative emissions.
- Created a time-series plot of emissions for those top industries.

📊 Output:
- **`top5_industries_emissions.png`**  
  A line graph showing GHG emission trends for the top 5 industries.

---

### 🤖 **Phase 2: Model Building & Evaluation**
- Encoded categorical columns: `Commodity Code`, `Description`, `Substance`, `Unit`.
- Trained a regression model using **XGBoost (XGBRegressor)**.
- Evaluated model using **R² Score** and **Mean Squared Error**.
- Saved predictions and trained model for reuse.
- Visualized model performance using a scatterplot of Actual vs. Predicted emissions.

📊 Output:
- - **`correlation_heatmap.png`** – Heatmap of numeric feature correlations.
- **`feature_importance.png`** – Bar chart of XGBoost feature importances.
- **`final_model.pkl`** – Serialized XGBoost model.
---

## 📈 Results (as of Phase 2)
- ✅ Model trained on cleaned emissions data.
- ✅ R² Score and MSE calculated to evaluate regression accuracy.
- ✅ Visual insights generated for actual vs. predicted performance.

---

## ⚙️ Tools & Technologies Used
- **Python**
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **XGBoost (XGBRegressor)** – Predictive modeling
- **Scikit-learn** – Data splitting, metrics, label encoding
- **Joblib** – Model serialization
- **VS Code** – Development environment

---