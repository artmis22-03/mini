import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the data from the CSV file
df = pd.read_csv('dataset.csv')  # Update with the actual path to your CSV file

# Fill missing values with a placeholder
df.fillna('none', inplace=True)

# Encode the categorical variables
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the logistic regression classifier
clf = LogisticRegression(random_state=42, max_iter=10000)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(clf, file)  
# Print the classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Function to classify a new test case
def classify_new_case(clf, label_encoders, new_case):
    # Create a DataFrame for the new case
    new_case_df = pd.DataFrame([new_case])
    
    # Encode the new case using the same encoders
    for column in new_case_df.columns:
        le = label_encoders[column]
        # Add new unseen labels to the classes
        unseen_labels = set(new_case_df[column].unique()) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.append(le.classes_, list(unseen_labels))
        new_case_df[column] = le.transform(new_case_df[column])
    
    # Predict the disease
    prediction = clf.predict(new_case_df)
    
    # Decode the prediction
    disease = label_encoders['Disease'].inverse_transform(prediction)
    return disease[0]

# Example test case
new_case = {
    "Symptom_1": "high_fever",
    "Symptom_2": "none",
    "Symptom_3": "none",
    "Symptom_4": "none",
    "Symptom_5": "none",
    "Symptom_6": "none",
    "Symptom_7": "none",
    "Symptom_8": "none",
    "Symptom_9": "none",
    "Symptom_10": "none",
    "Symptom_11": "none",
    "Symptom_12": "none",
    "Symptom_13": "none",
    "Symptom_14": "none",
    "Symptom_15": "none",
    "Symptom_16": "none",
    "Symptom_17": "none"
}

# Classify the new test case
predicted_disease = classify_new_case(clf, label_encoders, new_case)
print("\nPredicted Disease for the test case:", predicted_disease)
