import pandas as pd

#reading the file
df = pd.read_csv('used_cars.csv')

#removing unnecessary symbols such as "mi", "$", etc.
df['milage'] = df['milage'].str.replace('[, mi]','', regex=True)
df['price'] = df['price'].str.replace('[, $]','', regex=True)

print(df.head(1))
