import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading the data into the 'data' DataFrame
data = pd.read_csv("Ecommerce Customers.csv")
print(data.head())

# Separate the features (predictors) and the target variable
X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

print("Step 1: Data Separation - Features (X) and Target (y)")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the data into a training set and a test set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nStep 2: Data Splitting - Training and Testing Sets")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Create an instance of the linear regression model
linear_model = LinearRegression()


print("\nStep 3: Model Training")
# Train the linear regression model on the training set using the fit() method
linear_model.fit(X_train, y_train)


# Print the coefficients of the model
print("Coefficients:", linear_model.coef_)

# Test the model on the test set using the predict() method and store the predictions
y_pred = linear_model.predict(X_test)

print("\nStep 4: Model Testing")
print("Predicted values (y_pred):", y_pred)

