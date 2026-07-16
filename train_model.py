import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/healthcare_disease_dataset_10000.csv")

print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDisease Count:")
print(df["Disease"].value_counts())

print("\nDataset Information:")
print(df.info())

# Features aur Target ko alag karna

X = df.drop("Disease", axis=1)
y = df["Disease"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.preprocessing import LabelEncoder

# Gender Encoding
gender_encoder = LabelEncoder()
X["Gender"] = gender_encoder.fit_transform(X["Gender"])

print("\nGender Encoding Complete")
print(X["Gender"].head())

# Disease Encoding

disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(y)

print("\nDisease Encoding Complete")
print(y[:10])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("\n Model Training Completed")

from sklearn.metrics import accuracy_score

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

import joblib

# Save Model
joblib.dump(model, "model/disease_model.pkl")

# Save Label Encoders
joblib.dump(gender_encoder, "model/gender_encoder.pkl")
joblib.dump(disease_encoder, "model/disease_encoder.pkl")

print("\n✅ Model Saved Successfully")

print(X.columns.tolist())