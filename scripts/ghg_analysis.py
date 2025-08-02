import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Set paths
input_file = "data/emissions.csv"
visuals_dir = "visuals"
model_dir = "models"
os.makedirs(visuals_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(input_file)

# Cleaned the dataset by eliminating unnamed columns and handling missing values through row removal.
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.dropna(inplace=True)

print("✅ Data Loaded. Shape:", df.shape)

# Label encode categorical columns
label_encoders = {}
categorical_cols = ['Commodity Code', 'Description', 'Substance', 'Unit']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders in case needed later

# Define features and target
X = df.drop(columns=["Total Emissions"])
y = df["Total Emissions"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ Mean Squared Error: {mse:.4f}")

# Save predictions to CSV
results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
results_df.to_csv(os.path.join(visuals_dir, "xgb_predictions.csv"), index=False)

# Save the trained model
joblib.dump(model, os.path.join(model_dir, "final_model.pkl"))

# Plot Predictions vs Actual with diagonal reference line
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Actual", y="Predicted", data=results_df, alpha=0.7)
max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
plt.plot([0, max_val], [0, max_val], '--', color='red', label='Ideal Prediction Line')
plt.title(f"XGBoost Predictions vs Actual\nR² = {r2:.4f}")
plt.xlabel("Actual Emissions")
plt.ylabel("Predicted Emissions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "xgb_scatterplot.png"))
plt.show()

# Plot Feature Importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "feature_importance.png"))
plt.show()

# Plot Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "correlation_heatmap.png"))
plt.show()

# Plot Top 5 Industries Over Time (if columns exist)
if {'Description', 'Year', 'Total Emissions'}.issubset(df.columns):
    top5 = df.groupby("Description")["Total Emissions"].sum().nlargest(5).index
    df_top5 = df[df["Description"].isin(top5)]

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_top5, x="Year", y="Total Emissions", hue="Description", marker="o")
    plt.title("Top 5 US Industry GHG Emissions Over Time")
    plt.ylabel("Total Emissions (CO₂-equivalent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "top5_industries_emissions.png"))
    plt.show()

print("🎯 All done! Model, plots, and predictions saved successfully.")