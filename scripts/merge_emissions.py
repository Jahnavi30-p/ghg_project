import os
import pandas as pd

# List of filenames inside raw_data folder
file_names = [
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2010_Summary_Commodity).csv",
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2011_Summary_Commodity).csv",
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2012_Summary_Commodity).csv",
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2013_Summary_Commodity).csv",
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2014_Summary_Commodity).csv",
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2015_Summary_Commodity).csv",
    "SupplyChainEmissionFactorsforUSIndustriesCommodities(2016_Summary_Commodity).csv"
]

df_list = []

for filename in file_names:
    file_path = os.path.join("raw_data", filename)
    try:
        year = filename[-27:-23]  # Extracts the 4-digit year
        df = pd.read_csv(file_path)

        # Rename columns to expected names
        df = df.rename(columns={
            "Commodity Name": "Description",
            "Supply Chain Emission Factors without Margins": "Total Emissions"
        })

        # Add year column
        df["Year"] = int(year)

        df_list.append(df)
        print(f"✅ Loaded: {filename}")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")

# Combine all years
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)

    # Clean column names
    combined_df.columns = (
        combined_df.columns.str.strip()
        .str.replace('\n', ' ')
        .str.replace('  ', ' ', regex=False)
    )

    # Ensure required columns exist
    required = {"Description", "Total Emissions"}
    missing = required - set(combined_df.columns)
    if missing:
        raise ValueError(f"❌ Required columns missing after renaming: {missing}")

    # Save to 'data' folder
    os.makedirs("data", exist_ok=True)
    combined_df.to_csv("data/emissions.csv", index=False)
    print("✅ Merged file saved as 'data/emissions.csv'")
else:
    print("❌ No dataframes were loaded. Check your file paths.")