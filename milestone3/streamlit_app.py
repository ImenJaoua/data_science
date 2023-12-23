import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np
from ift6758.ift6758.client.serving_client import ServingClient
from ift6758.ift6758.client.game_Client import gameClient
from ift6758.ift6758.client import game_Client

#uncomment in case you are using local host 
#serving= ServingClient(ip="127.0.0.1", port=5000)

#uncomment in case you are using docker 
serving= ServingClient(ip="serving", port=5000)

client= gameClient()

st.title("Hockey Visualization App")

with st.sidebar:
    st.header("Input Parameters")
    workspace = st.text_input(label='workspace',value='imenjaoua')
    model = st.text_input(label='model',value='lr-distance')
    version = st.text_input(label='version',value='0.0.1')
    load = st.button("Load Model")
   

with st.container():
    st.header("Game ID Information")
    game_id=st.text_input(label="game_id",value=2021020329)
    ping=st.button("Ping Game")

with st.container():
    st.header("Predictions")
    if load:
        st.empty()  
        serving.download_registry_model(workspace,model,version)
        st.write("Model successfully downloaded! You can now proceed with predictions.")


register_g1=0
register_g2=0
register_id_game=0
homeOrAway=[]
with st.container():
    if ping: 
            
        teams = list(game_Client.team_names(str(game_id)))

        df_t1= client.pingGame(teams[0],str(game_id))        
        df_t2= client.pingGame(teams[1],str(game_id))   
            
        if len(df_t1)!=0 or len(df_t2)!=0:
            homeOrAway=[]
            homeOrAway.append(df_t1["homeOrAway"][1])
            homeOrAway.append(df_t2["homeOrAway"][1])
    
        if type(register_id_game)==type(0):
            register_id_game=str(game_id)
        if type(register_g1)==type(0):
            register_g1=df_t1
            register_g2=df_t2
        else:
            if register_id_game!=str(game_id):
  
                register_g1=df_t1
                register_g2=df_t2
                register_id_game=str(game_id)
            else:
                register_g1.append(df_t1,ignore_index=True)
                register_g2.append(df_t2,ignore_index=True)
            

            
        pt1 = register_g1.iloc[-1]["period"]
        timet1 = str(register_g1.iloc[-1]["periodTime"]).split(":")
        leftmn1 = 19-int(timet1[0])
        secondLeftt1=60-int(timet1[1])
        left_timet1= str(leftmn1)+":"+str(secondLeftt1)

        pt2 = register_g2.iloc[-1]["period"]
        timet2 = str(register_g2.iloc[-1]["periodTime"]).split(":")
        leftmn2 = 19-int(timet2[0])
        secondLeftt2=60-int(timet2[1])
        left_timet2= str(leftmn2)+":"+str(secondLeftt2)

        period=max([pt1,pt2])
        left_time=""
        if leftmn2<leftmn1:
            left_time=left_timet2
        elif leftmn1<leftmn2:
            left_time=left_timet1
        elif secondLeftt2<secondLeftt1:
            left_time=left_timet2
        elif secondLeftt1<secondLeftt2:
            left_time=left_timet1
        else:
            left_time=left_timet1

        XGt1=serving.predict(df_t1,str(game_id),teams[0])['predictions']
        XGt2=serving.predict(df_t2,str(game_id),teams[1])['predictions']
        sumt1 = sum(XGt1)
        sumt2= sum(XGt2)
        st.write(teams[0], " VS ", teams[1])
        st.write("period:",period)
        st.write("time left:",left_time)
        st.write("expected score for ",teams[0]," ",sumt1)
        st.write("expected score for ",teams[1]," ",sumt2)
        
  
        
        dff1 = pd.DataFrame({'Expected goals': XGt1})
        dff2 = pd.DataFrame({'Expected goals': XGt2})    
        new_df1 = pd.concat([register_g1, dff1], axis=1)[['Shot_distance','Shot_angle','Empty_net','period','Goal','Expected goals']]
        new_df2 = pd.concat([register_g2, dff2], axis=1)[['Shot_distance','Shot_angle','Empty_net','period','Goal','Expected goals']]
        new_df1['Empty_net'] = new_df1['Empty_net'].astype(int)
        new_df2['Empty_net'] = new_df2['Empty_net'].astype(int)

        st.write("Data of the team: ",teams[0])
        st.dataframe(new_df1)

        st.write("Data of the team: ",teams[1])
        st.dataframe(new_df2)


