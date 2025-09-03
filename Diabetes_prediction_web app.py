import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("D:/machine learning model/Diabetes_prediction_model/trained_model.sav", 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
       return 'The person is not diabetic'
    else:
       return 'The person is diabetic'
   
  #making user-interface  
   
def main():
      
      #giving a title for user-interface
      st.title('Diabetes prediction web app')
      
      #getting input data from the user
      Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
      
      Pregnancies = st.text_input('Number of pregnancies')
      Glucose=st.text_input('Glucose level')
      BloodPressure=st.text_input('Blood Pressure value')
      SkinThickness=st.text_input('Skin Thickness value')
      Insulin=st.text_input('Insulin level')
      BMI=st.text_input('BMI value')
      DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction value')
      Age=st.text_input('Age of the person')
      
      #code for prediction
      diagnosis=''
      
      #creating a button for prediction
      
      if st.button('Diabetes Test Result'):
          diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
      
      st.success(diagnosis)
      

if __name__ == '_main_':
    main()
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   