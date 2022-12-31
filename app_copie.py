

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#import lime as lime
filepath = "/Users/loicvalenti/Library/Mobile Documents/com~apple~CloudDocs/Formation Data Science/PROJET 7/Project-7-Credit-Score/Credit Application Results.csv"
client_predictions = pd.read_csv(filepath)
# Dash_App.py
### Import Packages ########################################
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pickle
### Setup ###################################################
app = dash.Dash(__name__)
app.title = 'Machine Learning Model Deployment'
server = app.server
filepath_database = "/Users/loicvalenti/Library/Mobile Documents/com~apple~CloudDocs/Formation Data Science/PROJET 7/train_features_removed.csv"
filepath_target = "/Users/loicvalenti/Library/Mobile Documents/com~apple~CloudDocs/Formation Data Science/PROJET 7/labels.csv"
database = pd.read_csv(filepath_database)
target = pd.read_csv(filepath_target)
database["TARGET"] = target["TARGET"]
### AGE GRAPH PLACEHOLDER ###################################################
age_data = pd.DataFrame()
age_data = database.loc[:,('TARGET', 'DAYS_BIRTH')]
age_data['YEARS_BIRTH'] = (age_data['DAYS_BIRTH'] / 365)

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_data['target'] = age_data['TARGET'] *100
age_data['bins'] = age_groups.index.astype(str)
fig =  px.bar(age_data, x =age_groups.index.astype(str) ,y = 'target')
#plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])


### load ML model ###########################################
### App Layout ###############################################
app.layout = html.Div([
    dcc.Input(id='client_id', value='Client ID', type = "number", min=2, max=1000000),
    
    html.Br(),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    
    html.Div(id='prediction output'),
    
    ], style = {'padding': '0px 0px 0px 150px', 'width': '50%'})
### Callback to produce the prediction #########################
@app.callback(
    Output('prediction output', 'children'),
    Input('client_id', 'value'))
   
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1,1]
        
        """
        Place holder for the graph printing methods and so on.
        """
        
        if prediction > 0.5000000:
            output = "Client's application was refused with {}% risk of defaulting".format(prediction*100)
        else:
            output =  "Client's application was accepted with {}% risk of defaulting".format(prediction*100)
                
    else:
        output = "Client's application is not in the database"
                    
    return f'{output}.'
### Run the App ###############################################
if __name__ == '__main__':
    app.run_server(debug=True)
