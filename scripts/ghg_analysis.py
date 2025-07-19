# Import Necessary Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
DATA_PATH = "data/emissions.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully from:", DATA_PATH)
except FileNotFoundError:
    print(f"❌ Dataset not found! Please make sure the file exists at: {DATA_PATH}")
    exit()

# Show Basic Info
print("\n📊 Dataset Info:")
print(df.info())

print("\n🔎 First 5 Rows:")
print(df.head())

print("\n📌 Column Names:")
print(df.columns)

# Check Required Columns
required_cols = {'Year', 'Total Emissions', 'Description'}
if not required_cols.issubset(df.columns):
    print(f"\n❌ Required columns missing: {required_cols - set(df.columns)}")
    exit()

# Create 'visuals' folder if it doesn't exist
if not os.path.exists("visuals"):
    os.makedirs("visuals")
    print("📁 'visuals/' folder created.")

# Convert 'Year' to int if not already
df['Year'] = df['Year'].astype(int)

# Plot GHG Emissions Over Time by Description
try:
    plt.figure(figsize=(14, 7))
    
    # Filter for top 5 industries with highest total emissions (cumulative)
    top_industries = df.groupby('Description')['Total Emissions'].sum().nlargest(5).index
    filtered_df = df[df['Description'].isin(top_industries)]

    sns.lineplot(data=filtered_df, x='Year', y='Total Emissions', hue='Description', marker='o')
    plt.title("Top 5 US Industry GHG Emissions Over Time", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Total Emissions (CO₂-equivalent)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = "visuals/top5_industries_emissions.png"
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Plot saved to {save_path}")

except Exception as e:
    print(f"⚠️ Could not plot graph: {e}")