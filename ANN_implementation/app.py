import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file :
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
##streamlit app
st.title("Customer Churn Prediction")

##user input
geography = st.selectbox(
    'Geography',
    onehot_encoder_geography.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

credit_score = st.number_input(
    'Credit Score',
    min_value=300,
    max_value=900,
    value=600
)

age = st.slider(
    'Age',
    min_value=18,
    max_value=92,
    value=40
)

tenure = st.slider(
    'Tenure',
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    'Balance',
    min_value=0.0,
    value=0.0
)

num_of_products = st.slider(
    'Number of Products',
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)

estimated_salary = st.number_input(
    'Estimated Salary',
)

##Prepare input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

