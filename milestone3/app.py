"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import pickle
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from comet_ml import API
from dotenv import load_dotenv
import numpy as np
from comet_ml import API
#import ift6758

# Load environment variables defined in .env file
load_dotenv('.env')

# Set the Comet API key
COMET_API_KEY = os.getenv('COMET_API_KEY')
print(COMET_API_KEY)
DEFAULT_VERSION = os.getenv('DEFAULT_VERSION')
DEFAULT_MODEL=os.getenv('DEFAULT_MODEL')
MODELS_DIR = os.getenv('MODELS_DIR', 'models/')

COMET_WORKSPACE = os.getenv('WORKSPACE')
COMET_REGISTRY_NAME = os.getenv('REGISTRY_NAME')  

available_model = None
available_model_name = None
def get_name(workspace=COMET_WORKSPACE, model=DEFAULT_MODEL, version=DEFAULT_VERSION):
    api = API(api_key=COMET_API_KEY)
    try:
        models = api.get_registry_model_names(workspace=COMET_WORKSPACE)
        if model not in models:
            raise ValueError(f"Model '{model}' not found in the registry.")
            
        details = api.get_registry_model_details(workspace, model, version)
        if details is None:
            raise ValueError(f"Details not found for model '{model}' and version '{version}'.")
        
        filename = details['assets'][0]['fileName']
        return filename
    except Exception as e:
        print(f"Error: {e}")
        return None
def get_model(filename, workspace=COMET_WORKSPACE, model=DEFAULT_MODEL, version=DEFAULT_VERSION, output_dir=MODELS_DIR):
    api = API(api_key=COMET_API_KEY)
    try:
        api.download_registry_model({workspace}, model, version, output_dir, expand=True)
        model_path = os.path.join(output_dir, filename)
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except ValueError:
        app.logger.info(str(ValueError))
        return None


def change_model(workspace, model, version,output_dir=MODELS_DIR):
    # TODO: check to see if the model you are querying for is already downloaded
    filename = get_name(workspace, model, version)
    is_downloaded = os.path.isfile(f'{MODELS_DIR}/{filename}')

    status_code = 200
    
    global available_model

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    if (is_downloaded):
        model_path = os.path.join(output_dir, filename)
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        app.logger.info('Model loaded from local repository')

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        model = get_model(workspace=workspace, model=model, version=version, filename=filename)
        if model:
            available_model = model
            app.logger.info('Model downloaded successfully from comet_ml')
        else:
            app.logger.info('Error occured while trying to download model comet_ml')
            status_code = 400

    global available_model_name
    available_model_name = model

    return status_code

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)


@app.before_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
 
    fileHandler = logging.FileHandler(LOG_FILE)
    fileHandler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(fileHandler)

    app.logger.info('Flask app started')
    filename = get_name()
    global available_model
    available_model = get_model(filename)
    global available_model_name
    available_model_name = DEFAULT_MODEL
    app.logger.info('Default model loaded')

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    app.logger.info(f'API call: /logs')
    
    status_code = 200
    with open(LOG_FILE, "r") as f:
        content = f.read()
  
    return jsonify(content), status_code  

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    app.logger.info(f'API call: /download_registry_model')
    json = request.get_json()
    app.logger.info(json)


    workspace = json['workspace']
    model = json['model']
    version = json['version']

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    status_code = change_model(workspace, model, version)
    
    # Build and log response
    response = {
            'message': 'Model loaded successfully' if status_code==200
            else 'Error occured while trying to download model from comet_ml'
        }

    app.logger.info(response)

    # return response and set status code
    return jsonify(response), status_code  # response must be json serializable!



@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    status_code = 200  

    dict_feat = {
    'lr-angle': ['angle'],
    'lr-distance': ['Shot_distance'],
    'lr-distance-angle': ['distance', 'angle']
    }
    try:
        app.logger.info('API call: /predict')

        json_data = request.get_json()
        app.logger.info(json_data)

        if available_model_name not in dict_feat:
            raise ValueError(f'Invalid model name: {available_model_name}')

        features = dict_feat[available_model_name]

        df = pd.DataFrame.from_dict(json_data)

        if not df.empty:
            features_data = df[features]

            preds = available_model.predict_proba(features_data)[:, 1]
        else:
            preds = np.array([])

        response = {
            'predictions': preds.tolist(),
            'status': 'success'
        }
    except ValueError as ve:
        status_code = 400  
        app.logger.error(f'ValueError prediction: {str(ve)}')
        response = {
            'message': 'Error in prediction. Invalid input.',
            'status': 'error'
        }
    except Exception as e:
        status_code = 500  # Internal Server Error
        app.logger.error(f'Error occurred in prediction: {str(e)}')
        response = {
            'message': 'Error in prediction. Internal server error.',
            'status': 'error'
        }

    app.logger.info(response)

    return jsonify(response), status_code

