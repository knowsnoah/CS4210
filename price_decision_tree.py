import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# reading the dataset 
df = pd.read_csv("clean_used_cars.csv")

#removing extreme outliers in price (top 1%)
upper = df["price"].quantile(0.99)
df = df[df["price"] <= upper]
# target variable
#using log1p to handle skewness in price
Y = np.log1p(df["price"])

#replacing model year with car age instead
df["car_age"] = 2025 - df["model_year"]

# mileage per year feature
df["mileage_per_year"] = df["mileage"] / (df["car_age"] + 1)

# define feature columns (raw, not yet encoded)
feature_cols = ["brand", "car_age", "mileage", "mileage_per_year", "clean_title",
                "fuel_type", "transmission_type", "model"]
X = df[feature_cols]

# split BEFORE encoding to avoid data leakage
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

# which columns are categorical / numeric
categorical_features = ["brand", "clean_title", "fuel_type", "transmission_type", "model"]
numeric_features = ["car_age", "mileage", "mileage_per_year"]

# set up column transformer
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# fit encoder on training data, transform both train and test
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

Y_test_real = np.expm1(Y_test) # inverse transform to get real prices for evaluation

# decision tree regressor
model = tree.DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, Y_train)

# predictions in log-space, then convert back to dollars     
Y_pred_log = model.predict(X_test)                           
Y_pred = np.expm1(Y_pred_log)                                

# evaluation in REAL dollars                                  
mse = mean_squared_error(Y_test_real, Y_pred)                 
r2 = r2_score(Y_test_real, Y_pred)                           

print("=== Decision Tree on log(price), evaluated in dollars ===") 
print(f"Mean Square Error: {mse:,.2f}")
print(f"R^2: {r2:.4f}")

#Trying a Gradient Boosting Regressor
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    random_state=42
)

model.fit(X_train, Y_train)
# predictions in log-space, then convert back to dollars      
Y_pred_log = model.predict(X_test)                            
Y_pred = np.expm1(Y_pred_log)                                 

# evaluation in REAL dollars                                   
mse = mean_squared_error(Y_test_real, Y_pred)                 
r2 = r2_score(Y_test_real, Y_pred)                            
print("=== HistGradientBoosting on log(price), evaluated in dollars ===")  
print(f"Mean Square Error (Gradient Boosting): {mse:,.2f}")
print(f"R^2 (Gradient Boosting): {r2:.4f}")
