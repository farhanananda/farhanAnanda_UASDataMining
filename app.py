import streamlit as st
import math
import pandas as pd 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
import pickle
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('Stroke Dataset.csv')

st.title('Aplikasi Informasi Tentang Penyakit Stroke')

html_layout1 = """
<br>
<div style="background-color:33186B ;border-radius: 25px;box-shadow: -5px -4px 3px rgba(32, 62, 27, 0.98); padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Stroke Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)
activities = ['XGB', 'Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?', activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>
    Stroke adalah kondisi medis yang terjadi ketika aliran darah ke otak terganggu. Hal ini dapat menyebabkan kematian sel-sel otak dan kerusakan otak</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('*Input Dataframe*')
    st.write(data)
    st.write('---')
    st.header('*Profiling Report*')
    st_profile_report(pr)

# Handling missing value dengan mean data
data['bmi'].fillna(math.floor(data['bmi'].mean()), inplace=True)

# Membuang coloum yg tidak di perlukan
data = data.drop(['id'], axis=1)

# Transformasi data
encode = LabelEncoder()
data['Residence_type'] = encode.fit_transform(data['Residence_type'].values)
data['gender'] = encode.fit_transform(data['gender'].values)
data['work_type'] = encode.fit_transform(data['work_type'].values)
data['ever_married'] = encode.fit_transform(data['ever_married'].values)
data['smoking_status'] = encode.fit_transform(data['smoking_status'].values)

# Train test split
X = data.drop('stroke', axis=1)
y = data['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# Create and train an XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Save the trained model to a new pickle file
with open('farhan.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    gender = st.sidebar.slider('Kelamin', 0, 200, 108)
    age = st.sidebar.slider('Usia', 0.8, 80.0, 1.0)
    hyper = st.sidebar.slider('Darah Tinggi', 0, 1, 0)
    heart = st.sidebar.slider('Sakit Jantung', 0, 1, 0)
    married = st.sidebar.slider('Status Nikah', 0, 1000, 120)
    work = st.sidebar.slider('Type Kerja', 0, 80, 25)
    residence = st.sidebar.slider('Residence Type', 0.05, 2.5, 0.45)
    gula = st.sidebar.slider('Gula Darah', 77.0, 280.0, 77.0)
    bmi = st.sidebar.slider('BMI', 20.0, 80.0, 50.0)
    smoke = st.sidebar.slider('Type Perokok', 0, 3, 1)
    user_report_data = {
        'gender': gender,
        'age': age,
        'hypertension': hyper,
        'heart_disease': heart,
        'ever_married': married,
        'work_type': work,
        'Residence_type': residence,
        'avg_glucose_level': gula,
        'bmi': bmi,
        'smoking_status': smoke
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

# Load the trained XGBoost model
xgb_model = pickle.load(open('XGB.pkl', 'rb'))

user_result = xgb_model.predict(user_data)
xgb_score = accuracy_score(y_test, xgb_model.predict(X_test))

# Output
st.subheader('Hasilnya adalah : ')
output = ''
if user_result[0] == 0:
    output = 'Kamu Aman'
else:
    output = 'Kamu terkena stroke'
st.title(output)
st.subheader('Model yang digunakan : \n' + option)
st.subheader('Accuracy : ')
st.write(str(xgb_score*100)+'%')