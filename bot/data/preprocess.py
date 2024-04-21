import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Load the training data
data = pd.read_csv("training_data.csv")

# Convert 'timestamp' column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

# # Extract month and day information from the timestamp
# data['month'] = data['timestamp'].dt.month
# data['day'] = data['timestamp'].dt.day
# data['hour'] = data['timestamp'].dt.hour


filtered_df = data[(data['timestamp'].dt.month == 4) | (data['timestamp'].dt.month == 5)]

X = filtered_df.dropna()

# Separate features and target variable (if needed)
# X = X.drop(columns=["timestamp"
#                     # "pv_power_forecast_1h",
#                     # "pv_power_forecast_2h",
#                     # "pv_power_forecast_24h",
#                     # "pv_power_basic",
#                     # "pv_power_basic_forecast_1h",
#                     # "pv_power_basic_forecast_2h",
#                     # "pv_power_basic_forecast_24h"
#                     ])  # Remove timestamp column as it's not needed for imputation


X.to_csv("pruned_training_mayril.csv", index=False)
# regressor = RandomForestRegressor(
#         # We tuned the hyperparameters of the RandomForestRegressor to get a good
#         # enough predictive performance for a restricted execution time.
#         n_estimators=50,
#         max_depth=10,
#         bootstrap=False,
#         # max_samples=0.5,
#         n_jobs=4,
#         random_state=0,
#     )

# # Initialize the IterativeImputer with RandomForestRegressor as estimator
# imputer = IterativeImputer(
#         estimator=regressor, 
#         random_state=0, 
#         max_iter=20, 
#         verbose=2
#     )

# # Fit and transform the data to impute missing values
# X_imputed = imputer.fit_transform(X)

# # Convert the imputed array back to DataFrame with column names
# X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Save the imputed DataFrame to a new CSV file
# X_imputed_df.to_csv("imputed_training_v3.csv", index=False)

# Optional: Display summary of imputation (if needed)
# print("Imputation summary:")
# print(X_imputed_df.isnull().sum())  # Check if there are any remaining missing values