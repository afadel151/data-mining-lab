import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Bacteria_dataset_Multiresictance.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -----------------------------
# 2. Cleaning
# -----------------------------

# Clean Souches (species names)
df['Souches'] = df['Souches'].str.replace(r"S\d+", "", regex=True)
df['Souches'] = df['Souches'].str.strip().str.title()

# Identify antibiotic columns (exclude metadata)
exclude_cols = ['Souches', 'Age/Gender', 'Diabetes', 'Hypertension', 
                'Hospital', 'Infection_Freq', 'Name', 'Email', 'Address', 
                'Notes', 'Collection Date']
antibiotic_cols = [col for col in df.columns if col not in exclude_cols]

# Normalize antibiotic values
for col in antibiotic_cols:
    df[col] = df[col].astype(str).str.upper().replace({
        "INTERMEDIATE": "I",
        "MISSING": np.nan,
        "?": np.nan,
        "-": np.nan
    })
    df[col] = df[col].replace({"R": "R", "S": "S", "I": "I"})

# Split Age/Gender
df[['Age', 'Gender']] = df['age/gender'].str.split("/", expand=True)
df.drop(columns=['age/gender'], inplace=True)

# Clean Age
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 120), 'Age'] = np.nan

# Clean Gender
df['Gender'] = df['Gender'].str.upper().replace({"FEMALE":"F", "MALE":"M"})

# Risk factors
risk_cols = ['Diabetes', 'Hypertension', 'Hospital_before']
for col in risk_cols:
    df[col] = df[col].astype(str).str.lower().replace({
        "yes": "Yes", "y": "Yes", "1": "Yes", "true": "Yes",
        "no": "No", "n": "No", "0": "No", "false": "No",
        "nan": np.nan
    })

# Infection Frequency
mapping = {"never":0, "rarely":1, "regularly":2, "often":3}
df['Infection_Freq'] = df['Infection_Freq'].astype(str).str.lower().map(mapping)

# Drop irrelevant columns
df.drop(columns=['Name', 'Email', 'Address', 'Notes', 'Collection Date'], 
        errors="ignore", inplace=True)

# Remove duplicates
df = df.drop_duplicates()

# -----------------------------
# 3. MultiResistance Indicator
# -----------------------------
families = {
    "Beta-lactams": ["AMX/AMP", "PEN", "OXA"],
    "Quinolones": ["CIP", "NOR"],
    "Aminosides": ["GEN", "TOB"],
    # Add more families if your dataset includes them
}

def is_multiresistant(row):
    resistant_families = 0
    for fam, cols in families.items():
        if any((c in row.index and row[c] == "R") for c in cols):
            resistant_families += 1
    return 1 if resistant_families >= 3 else 0

df['MultiResistance'] = df.apply(is_multiresistant, axis=1)

# Save cleaned dataset
df.to_csv("Bacteria_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as Bacteria_cleaned.csv")

