import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Display the decision tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print("Decision Tree Rules:\n")
print(tree_rules)

# Predict on the test set
y_pred = clf.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=label_encoders['Disease'].classes_)
print("\nClassification Report:\n")
print(report)

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
    "Symptom_1": "itching",
    "Symptom_2": "vomiting",
    "Symptom_3": "weight_loss",
    "Symptom_4": "fatigue",
    "Symptom_5": "high_fever",
    "Symptom_6": "yellowish_skin",
    "Symptom_7": "abdominal_pain",
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

# Load the symptom description data
df1 = pd.read_csv("symptom_Description.csv")
dfdict = df1.set_index('Disease').T.to_dict('list')

# Print the description of the predicted disease
print(dfdict[predicted_disease][0])

# Debug: Verify the encoding of the new case
# encoded_new_case = {}
# for column in new_case:
#     encoded_new_case[column] = label_encoders[column].transform([new_case[column]])[0]
# print("\nEncoded New Case:", encoded_new_case)
