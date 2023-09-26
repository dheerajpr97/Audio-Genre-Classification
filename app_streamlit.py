# Description: This file contains the code for the streamlit app.

import os
import random
from cnnClassifier.pipeline.predict import PredictionPipeline
import streamlit as st

os.putenv('LANG', 'en_US.UTF-8') 
os.putenv('LC_ALL', 'en_US.UTF-8') 


class ClientApp: 
    def __init__(self):
        pass
        
# Function to predict the genre of the audio file uploaded by the user passed as an argument to the PredictionPipeline class
def predict(audio):
    '''
    Function to predict the genre of the audio file uploaded by the user passed as an argument to the PredictionPipeline class
    Input: audio file
    Output: predicted genre
    '''    
    classifier = PredictionPipeline(audio)
    result = classifier.predict()
    
    return result

def main():
    #st.title("Audio Genre Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Audio Genre Classification ML App </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload Audio", type=['wav'])
    if audio_file is not None:
        st.success("File Uploaded Successfully")
        if st.button("Predict"):
            result = predict(audio_file)
            st.success(f"The song is predicted to be in the {result} genre.")

    
if __name__ == "__main__":
    main()
