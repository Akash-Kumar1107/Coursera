import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Ecommerce Customers.csv")
print(data.head())

# Get statistical information about the dataset
data_description = data.describe()
print(data_description)

# Get information about the data types
data_info = data.info()
print(data_info)

# Assuming you have already loaded the data into the 'data' DataFrame

# Use seaborn's jointplot with kind='reg'
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=data, kind='reg')

# Use seaborn's jointplot with kind='reg'
sns.jointplot(x='Time on App', y='Length of Membership', data=data, kind='reg')

# Show the plot
plt.show()

# Use seaborn's pairplot to visualize relationships between all numerical variables
sns.pairplot(data)

# Show the plot
plt.show()

# Graph a linear model plot using x = length of membership and y= yearly amount spent. (Use seaborn)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=data)

# Show the plot
plt.show()

