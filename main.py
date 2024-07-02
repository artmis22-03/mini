import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the models
with open('decision_tree_model.pkl', 'rb') as file:
    clf = pickle.load(file)

with open('knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

# Load the encoders
df = pd.read_csv('dataset.csv')
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Load the symptom description data
df1 = pd.read_csv("symptom_Description.csv")
dfdict = df1.set_index('Disease').T.to_dict('list')

# Function to classify a new test case
def classify_new_case(model, label_encoders, new_case):
    new_case_df = pd.DataFrame([new_case])
    for column in new_case_df.columns:
        le = label_encoders[column]
        unseen_labels = set(new_case_df[column].unique()) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.append(le.classes_, list(unseen_labels))
        new_case_df[column] = le.transform(new_case_df[column])
    prediction = model.predict(new_case_df)
    disease = label_encoders['Disease'].inverse_transform(prediction)
    return disease[0]

# Streamlit UI
st.title('Disease Prediction from Symptoms')

# Create input fields for symptoms
symptoms = {}
# for i in range(1, 18):
#     symptom = st.text_input(f'Symptom {i}', 'none')
#     symptoms[f'Symptom_{i}'] = symptom
all=st.text_input('Symptoms', 'none')
alls=all.split(',')
for i in range(1,18):
    symptoms[f'Symptom_{i}'] = 'none'
    
for i in range(1, len(alls)):
    symptoms[f'Symptom_{i}'] = alls[i-1]
# Dropdown to choose model
model_choice = st.selectbox('Choose the model', ('Decision Tree', 'KNN'))

# Predict button
if st.button('Predict Disease'):
    if model_choice == 'Decision Tree':
        predicted_disease = classify_new_case(clf, label_encoders, symptoms)
    else:
        predicted_disease = classify_new_case(knn, label_encoders, symptoms)
    
    description = dfdict[predicted_disease][0]
    
    st.subheader('Predicted Disease')
    st.write(predicted_disease)
    
    st.subheader('Disease Description')
    st.write(description)
