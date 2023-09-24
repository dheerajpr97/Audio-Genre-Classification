from flask import Flask, request, jsonify, render_template
import os
import random
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage, decodeAudio
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
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system('dvc repro') #("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    
    audio = request.files["UploadedAudio"]
    # random string of digits for file name
    file_name = str(random.randint(0, 100000))

    # save the file locally
    audio.save(file_name)
    
    classifier = PredictionPipeline(file_name)
    result = classifier.predict()
    
    os.remove(file_name)
    # message to be displayed on the html webpage
    prediction_message = f"""
    The song is predicted to be in the {result} genre.
    """
    return render_template("index.html", prediction_text=prediction_message)

if __name__ == "__main__":
    #clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    # app.run(host='0.0.0.0', port=80) #for AZURE