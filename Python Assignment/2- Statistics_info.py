import pandas as pd

# Loading the data into the 'data' DataFrame
data = pd.read_csv("Ecommerce Customers.csv")
print(data.head())

# Get statistical information about the dataset
data_description = data.describe()
print(data_description)

# Get information about the data types
data_info = data.info()
print(data_info)