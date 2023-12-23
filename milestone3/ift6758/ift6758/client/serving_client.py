import json
import requests
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class ServingClient:
    def __init__(self, ip: str = "localhost", port: int = 5000, features=None):
        if not (os.path.isfile('predicted.json') and os.access('predicted.json', os.R_OK)):
            with open('predicted.json', 'w') as outfile:
                data = {}
                json.dump(data, outfile)
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["Shot_distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame, id_game, team) -> list:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
            id_game: to track the already predicted events
        """

        url = self.base_url+"/predict"

        response_API = requests.post(url, json=json.loads(X.to_json()))
        print('worked: ',response_API)
        prediction=json.loads(response_API.text)



        f = open('predicted.json')
        preds = json.load(f)

        if (str(id_game) in preds.keys()) and (team in preds[id_game].keys()):
            recent = eval(preds[id_game][team])
            recent.update(prediction)
            preds[id_game].update({team: str(recent)})


        elif (str(id_game) in preds.keys()):

            preds[id_game][team]= str(prediction)

        else:
            preds.update({id_game: {team: str(prediction)}})

        with open('predicted.json', 'w') as outfile:
            json.dump(preds, outfile)
        return eval(preds[id_game][team])



    def logs(self) -> dict:
        """Get server logs"""
        endpoint = f"{self.base_url}/logs"
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:
        https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        data={
            'workspace':workspace,
            'model': model,
            'version':version
        }
        
        response_API = requests.post(self.base_url+"/download_registry_model", json=data)
        return json.loads(response_API.text)
#if __name__ == "__main__":

#    Client=ServingClient("localhost",5000)