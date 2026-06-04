# ==============================
# Rolling Resistance Prediction Study
# Machine Learning Model
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("DATA.csv")

print("Dataset Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())

# ==============================
# 2. Data Preprocessing
# ==============================

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data["BRAND_LE"]=label_encoder.fit_transform(data['BRAND'])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data["SIZE_LE"]=label_encoder.fit_transform(data['SIZE'])

data.drop('SIZE',axis=1,inplace=True)
data.drop('BRAND',axis=1,inplace=True)

data.dtypes

# checking null values

null_values = data.isnull().sum()
print(null_values)

# ==============================
# 3. Features & Target
# ==============================
target_column = "RR"   # change this, if your dataset uses different name

X = data.drop(target_column, axis=1)
y = data[target_column]

# ==============================
# 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# adjust test_size and random_state as per your requirement

# ==============================
# 5. Linear Regression
# ==============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

print("\n===== Linear Regression =====")
print("R² Score:", r2_lr)
print("MAPE:", mape_lr)

# ==============================
# 6. Random Forest
# ==============================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

print("\n===== Random Forest =====")
print("R² Score:", r2_rf)
print("MAPE:", mape_rf)

# ==============================
# 7. Feature Importance
# ==============================
feature_importance = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\n===== Feature Importance =====")
print(feature_importance_df)

# Plot and save for analsyis
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

# ==============================
# 8. Model Comparison
# ==============================
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [r2_lr, r2_rf],
    "MAPE": [mape_lr, mape_rf]
})

print("\n===== Model Comparison =====")
print(results)

# ==============================
# END
# ==============================
