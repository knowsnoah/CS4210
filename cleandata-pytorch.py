import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

df = pd.read_csv('used_cars.csv')

for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()

def simply_transmission(s):
    if not isinstance(s, str):
        return None
    s = s.lower()
    if "manual" in s or "m/t" in s:
        return "Manual"
    elif "cvt" in s:
        return "CVT"
    elif 'automatic' in s or "a/t" in s:
        return "A/T"
    else:
        return "Other"

mileage = df['mileage'].str.replace('[, mi]','', regex=True).astype(float)
price = df['price'].str.replace('[, $]','', regex=True).astype(float)

transmission_type = df['transmission'].apply(simply_transmission)

model = df['model']

model_year = df['model_year'].astype(float)

fuel_type = df['fuel_type'].fillna('Other').replace('-', 'Other')

bad_tokens = ['-', '–', '—', '', 'nan', 'None', 'not supported']

fuel_type = (
    df['fuel_type'].astype('string').str.strip().replace(bad_tokens, pd.NA).fillna('Other'))

clean_title = df['clean_title'].fillna('No').str.strip().str.title().map({"Yes": 1, "No": 0}).astype(float)

encoder = LabelEncoder()
brands_encoded = encoder.fit_transform(df['brand'].fillna("Unknown"))

encoder = LabelEncoder()
models_encoded = encoder.fit_transform(df['model'].fillna("Unknown"))

encoder = LabelEncoder()
fuel_type_encoded = encoder.fit_transform(fuel_type.fillna("Unknown"))

encoder = LabelEncoder()
transmission_type_encoded = encoder.fit_transform(transmission_type.fillna("Unknown"))

encoder = LabelEncoder()
clean_title_encoded = encoder.fit_transform(clean_title.fillna("Unknown")).astype(float)

scaler = StandardScaler()
mileage_scaled = scaler.fit_transform(mileage.values.reshape(-1, 1)).flatten()

scaler = StandardScaler()
model_year_scaled = scaler.fit_transform(model_year.values.reshape(-1, 1)).flatten()

log_price = np.log(price.clip(lower=1.0))

data = {
    'brand': brands_encoded,
    'model': models_encoded,
    'model_year': model_year_scaled,
    'mileage': mileage_scaled,
    'fuel_type': fuel_type_encoded,
    'transmission_type': transmission_type_encoded,
    'clean_title': clean_title_encoded,
    'price': price,
    'log_price': log_price
}

df_new = pd.DataFrame(data)

df_new.to_csv('pytorch_used_cars.csv', index=False)
print("New simplified Databased saved as: 'pytorch_used_cars.csv'")