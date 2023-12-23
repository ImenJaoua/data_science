# -*- coding: utf-8 -*-
import os
import math 
import requests
import pandas as pd
import logging
import numpy as np
import json
import json
logger = logging.getLogger(__name__)

if not (os.path.isfile('suivi.json') and os.access('suivi.json', os.R_OK)):
    with open('suivi.json', 'w') as outfile:
        data = {}
        json.dump(data, outfile)


def team_names(id_game):
    fdata = requests.get("https://api-web.nhle.com/v1/gamecenter/"+id_game+"/play-by-play/")
    plays_data =json.loads(fdata.text)
    teams = set()
    teams.add(plays_data["homeTeam"]["name"]["default"])
    teams.add(plays_data["awayTeam"]["name"]["default"])

    return teams

#Fonction pour déterminer si le filet est vide
def is_empty_net(situation_code, event_owner_team_id, home_team_id, away_team_id):
    # Obtenez le premier chiffre du situation_code
    first_digit = int(str(situation_code)[0])
    last_digit=int(str(situation_code)[3])
    # Déterminez si le filet est vide
    return not(first_digit  if event_owner_team_id == away_team_id else last_digit)

def homeoraway( event_owner_team_id, home_team_id, away_team_id):
    return not("away"  if event_owner_team_id == away_team_id else "home")

def euclidean_distance(point1, point2):
    return math.sqrt(math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2))
    

def extractFeatures(fdata,id_game,team_Shooter,idx=0):
    try:
        with open('suivi.json') as f:
            data = json.load(f)
        idx=int(data[id_game][team_Shooter])
    except:
        idx=0

    fullGame=fdata
    allgame = fullGame
    plays_data=allgame['plays']
    if len(plays_data) == 0:
        return np.nan
    
    home_team_id = allgame["homeTeam"]["id"]
    away_team_id = allgame["awayTeam"]["id"]
    data=[]
    if team_Shooter==allgame["homeTeam"]["name"]["default"]:
        ts_id=home_team_id
    else:
        ts_id=away_team_id
    for play in plays_data:
        if 'shot' in play.get("typeDescKey").lower() or 'goal' in play.get("typeDescKey").lower():
            if play.get("details", {}).get("eventOwnerTeamId")==ts_id:
                x_coord = play["details"]["xCoord"] if "xCoord" in play["details"].keys() else 0
                y_coord = play["details"]["yCoord"] if "yCoord" in play["details"].keys() else 0
                
                x=x_coord
                y=y_coord


                # Déterminer les buts 
                is_goal=1 if play["typeDescKey"] == "goal" else 0

                # Déterminer si le filet est vide
                empty_net=is_empty_net(play.get("situationCode"), play.get("details", {}).get("eventOwnerTeamId"), home_team_id, away_team_id)
                awho=homeoraway(play.get("details", {}).get("eventOwnerTeamId"), home_team_id, away_team_id)
                period=play.get("period")
                periodtime=play.get('timeInPeriod')
                data.append([x,y,is_goal,empty_net,period,awho,periodtime])

    columns=["X","Y","Goal","Empty_net","period","homeOrAway","periodTime"]
    df = pd.DataFrame(data, columns=columns)  
    df.Empty_net = df.Empty_net.fillna(False) 
        
    df['Avg'] = df.groupby(['period'])['X'].transform('mean')
    def distance_x_from_net(row):
      x = row.X
      x_net = 89 if row.Avg > 0 else -89
      return abs(x_net-x)

    df['X_net'] = df.apply(distance_x_from_net, axis=1)
    df['Shot_distance'] = df.apply(lambda row: euclidean_distance([row.X_net, row.Y], [0, 0]), axis=1)

    df['Shot_angle'] = df.apply(lambda row: math.degrees(math.atan2(abs(row.Y), row.X_net)), axis=1)

    sign = np.sign(df.Y) * np.sign(df.X)
    df['Shot_distance'] *= sign
    df.drop(['Avg', 'X_net', 'X', 'Y'], axis=1, inplace=True)
    return df
        
class gameClient:
    def __init__(self):

        logger.info(f"Initializing ClientGame; base URL: ")

    def pingGame(self, team,id_game="2021020329",idx=0) :
        fdata = requests.get("https://api-web.nhle.com/v1/gamecenter/"+id_game+"/play-by-play/")
        dtf=extractFeatures(json.loads(fdata.text),id_game,team,idx=idx)
        return dtf


    
    
    
   