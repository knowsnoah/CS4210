CS4210 – Machine Learning Project
Used Car Price Predictor Model

Team Members -

David Carbajal – djcarbajal@cpp.edu
Noah Ojeda – nsojeda@cpp.edu
David Malone – davidmalone@cpp.edu
Ethan Owusu-Bour – eyowusubour@cpp.edu

Project Overview - 

This project builds a machine learning model that predicts the price of a used car based on various features such as year, mileage, brand, transmission type, and fuel type.
We used Python (Pandas, Scikit-learn) to preprocess the dataset and train regression models.

Dataset - 

Source: used_cars.csv (original raw dataset)

Cleaned file: clean_used_cars.csv

Final features used:

model_year, mileage, fuel_type, brand, model, transmission_type, clean_title, price (target variable)

Notes on preprocessing - 

Removed symbols like $, ,, and mi from numerical columns.

Encoded brand using LabelEncoder and stored the brand mapping in brand_map.txt.

Standardized transmission_type (Manual, CVT, Automatic, Other).

Cleaned fuel_type values and replaced invalid ones (e.g. "–", "not supported") with "Other".

Converted clean_title to binary: 1 = Yes, 0 = No.

Removed missing and inconsistent entries to ensure model stability.

