import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# import lime as lime
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

# import pickle
### Setup ###################################################
app = dash.Dash(__name__)
app.title = 'Machine Learning Model Deployment'

server = app.server
colors = {
    'background': '#111111',
    'text': '#FFFBF6'
}
filepath_database = "Project-7-Credit-Score/Dataset_for_webapp.csv"

filepath_age_groups = "Project-7-Credit-Score/age_groups.csv"
filepath_target = "Project-7-Credit-Score/labels.csv"
target = pd.read_csv(filepath_target)
database = pd.read_csv(filepath_database)

database["TARGET"] = target["TARGET"]

age_groups = pd.read_csv(filepath_age_groups)
### load ML model ###########################################
### App Layout ###############################################
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Div(children='Interactive Client Application Reviewer', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Input(id='client_id', value='Client ID', type="number", min=2, max=1000000, style={
        'textAlign': 'left',
        'color': colors['background']
    }),

    html.Div(id='prediction_output', style={
        'textAlign': 'left',
        'color': colors['text']
    }),

    dcc.Graph(id='Age_group_graph'),

    html.H4("Analysis of EXT_SOURCE_1's effect"),
    html.P("Select Distribution:"),
    dcc.RadioItems(
        id='distribution',
        options=['box', 'violin', 'rug'],
        value='box', inline=True
    ),

    dcc.Graph(id='EXT_SOURCE_1'),

])


### Callback to produce the prediction #########################
@app.callback(
    Output('prediction_output', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1, 1]

        """
        Place holder for the graph printing methods and so on.
        """

        if prediction > 0.5000000:
            output = "Client's application was refused with {}% risk of defaulting".format(prediction * 100)
        else:
            output = "Client's application was accepted with {}% risk of defaulting".format(prediction * 100)

    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('Age_group_graph', "figure"),
    Input('client_id', 'value'))
def show_client_position_Age_group_graph(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        x_ticks_labels = ["(20.0, 25.0", "(25.0, 30.0]", "(30.0, 35.0]", "(35.0, 40.0]", "(40.0, 45.0]", "(45.0, 50.0]",
                          "(50.0, 55.0]", "(55.0, 60.0]", "(60.0, 65.0]", "(65.0, 70.0]"]
        fig = px.bar(x=x_ticks_labels, y=100 * age_groups['TARGET'],
                     title="Failure to Repay by Age Group",
                     labels={
                         "y": "Failure to Repay (%)"
                     })

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        fig.add_vline(
            x=round(database.loc[client_id, "DAYS_BIRTH"] * 69.120548) % 10,
            # reverse engineered value from feature engineering
            line_width=3, line_dash="dash",
            line_color="red")
        return fig
    else:

        x_ticks_labels = ["(20.0, 25.0", "(25.0, 30.0]", "(30.0, 35.0]", "(35.0, 40.0]", "(40.0, 45.0]", "(45.0, 50.0]",
                          "(50.0, 55.0]", "(55.0, 60.0]", "(60.0, 65.0]", "(65.0, 70.0]"]
        fig = px.bar(x=x_ticks_labels, y=100 * age_groups['TARGET'],
                     title="Failure to Repay by Age Group",
                     labels={
                         "y": "Failure to Repay (%)"
                     })

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'])
        return fig


"""
@app.callback(
    Output('EXT_SOURCE_1', "figure"),
    Input('client_id', 'value'))
def show_client_position_EXT_SOURCE_1(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig = px.histogram(database["EXT_SOURCE_1"],
                           title="EXT SOURCE 1")

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        fig.add_vline(
            x=database.loc[client_id, "EXT_SOURCE_1"],
            line_width=3, line_dash="dash",
            line_color="red")
        return fig
    else:
        fig = px.line(database, x = database[""]
                      title="EXT SOURCE 1")

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        return fig
"""


@app.callback(
    Output("EXT_SOURCE_1", "figure"),
    Input("distribution", "value"))
def display_graph(distribution):
    fig = px.histogram(
        database, x="EXT_SOURCE_1",
        marginal=distribution,
        hover_data=database.columns)
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig


### Run the App ###############################################
if __name__ == '__main__':
    app.run_server(debug=True)
