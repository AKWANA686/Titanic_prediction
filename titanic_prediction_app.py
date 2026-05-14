import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)

sex = 0 if sex == "male" else 1

input_data = pd.DataFrame(
    {
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
    }
)

prediction = model.predict(input_data)

if prediction[0] == 1:
    st.success("Passenger Survived")
else:
    st.error("Passenger Did Not Survive")


