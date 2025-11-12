import pandas as pd
from sklearn import tree 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#reading the dataset 
df = pd.read_csv("clean_used_cars.csv")

#making the price ranges and labels for classification:
#ADD HERE

#defining X and Y
X = df.drop('price', axis=1)
Y = df['price']

#setting up the encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

#categories that need to be encoded: fuel_type, transmission_type, model
fuel_type_encoded = encoder.fit_transform(X[["fuel_type"]])
transmission_type_encoded = encoder.fit_transform(X[["transmission_type"]])
model_encoded = encoder.fit_transform(X[["model"]])

#combining all features into X
X = np.concatenate((
    df[["brand", "model_year", "mileage", "clean_title"]].to_numpy(),
    fuel_type_encoded,
    transmission_type_encoded,
    model_encoded
), axis=1)

#splitiing the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

#finish the decision tree model
model = tree.DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, Y_train)

#predicting the test set using the model
Y_prediction = model.predict(X_test)

#evaluating the model
mse = mean_squared_error(Y_test, Y_prediction)
r2 = r2_score(Y_test, Y_prediction)

print(f"Mean Square Error: {mse:,.2f}")
print(f"R^2:  {r2:.4f}")