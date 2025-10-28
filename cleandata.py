import pandas as pd
from sklearn.preprocessing import LabelEncoder
#going to simplfiy the used_cars.csv file into these columns:
#brand (encoded), model, model_year, mileage, fuel_type, transmission_type, clean_title, price

df = pd.read_csv('used_cars.csv')

#strip whitesapce from the dataset
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()

#function for reading/simplifing the transmission
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

#removing unnecessary symbols such as "mi", "$", etc.
mileage = df['mileage'].str.replace('[, mi]','', regex=True).astype(float)
price = df['price'].str.replace('[, $]','', regex=True).astype(float)

#storing the new transmission types
transmission_type = df['transmission'].apply(simply_transmission)

#getting model
model = df['model']

#getting_model years
model_year = df['model_year']

#getting fuel_type
fuel_type = df['fuel_type']

#tunring clean_title into true(1) or false(1) instead of Yes and No 
clean_title = df['clean_title'].fillna('No').str.strip().str.title().map({"Yes": 1, "No": 1})

#encoding the models
encoder = LabelEncoder()
brands_encoded = encoder.fit_transform(df['brand'].fillna("Unknown"))

# combine them into a new DataFrame
data = {
    'brand': brands_encoded,
    'model': model,
    'model_year': model_year,
    'mileage': mileage,
    'fuel_type': fuel_type,
    'transmission_type': transmission_type,
    'clean_title': clean_title,
    'price': price
}

df_new = pd.DataFrame(data)

# show the first few rows
print(df_new.head())

# save to a new CSV file
df_new.to_csv('clean_used_cars.csv', index=False)
print("New simplified Databased saved as: 'used_cars_database.csv'")

#to see the brand mapping
brand_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(f'Brand map: {brand_map}')