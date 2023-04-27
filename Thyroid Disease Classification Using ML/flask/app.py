import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("thyroid_dataset.csv")

# Preprocess data
data = data.dropna()  # remove missing values
data = pd.get_dummies(data, columns=["sex", "on_thyroxine", "query_on_thyroxine", "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre", "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured"])  # one-hot encoding
X = data.drop(columns=["Class_thyroid"])  # features
y = data["Class_thyroid"]  # labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
