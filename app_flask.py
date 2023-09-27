from flask import Flask, request, jsonify, render_template
import os
import random
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        pass
        
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html') # render the index.html page

@app.route("/train", methods=['GET','POST']) 
@cross_origin()
def trainRoute():
    os.system('dvc repro') #("python main.py") # run the main.py file to train the model or run the dvc repro command to verify the pipeline
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():    
    audio = request.files["UploadedAudio"] # get the audio file from the request    
    file_name = str(random.randint(0, 100000)) # random string of digits for file name
    audio.save(file_name) # save the audio file temporarily   
    classifier = PredictionPipeline(file_name) # create a classifier object with the audio file
    result = classifier.predict() # predict the genre of the audio file   
    os.remove(file_name) # remove the temp audio file from the server
    
    prediction_message = f"""
    The song is predicted to be in the {result} genre.
    """
    return render_template("index.html", prediction_text=prediction_message)

if __name__ == "__main__":
    #clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) # local host
    #app.run(host='0.0.0.0', port=8080) # for AWS
    