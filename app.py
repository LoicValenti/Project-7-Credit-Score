import pandas as pd
# import matplotlib.pyplot as plt
import plotly.express as px

# import plotly.figure_factory as ff

# import lime as lime
filepath = "Credit Application Results.csv"
client_predictions = pd.read_csv(filepath)
# Dash_App.py
### Import Packages ########################################
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

# import dash_bootstrap_components as dbc
# import numpy as np

# import pickle
### Setup ###################################################
app = dash.Dash(__name__)
app.title = 'Machine Learning Model Deployment'

server = app.server
colors = {
    'background': '#111111',
    'text': '#FFFBF6'
}
filepath_database = "Dataset_for_webapp.csv"

filepath_age_groups = "age_groups.csv"
filepath_target = "Credit Application Results.csv"
target = pd.read_csv(filepath_target)
database = pd.read_csv(filepath_database)
target.loc[target["TARGET"] > 0.5, "TARGET"] = 1
target.loc[target["TARGET"] <= 0.5, "TARGET"] = 0
database["SK_ID_CURR"] = [i for i in target["SK_ID_CURR"]]
database = database.set_index(database["SK_ID_CURR"]).drop(columns=["SK_ID_CURR"])
database.insert(1, "TARGET", target["TARGET"])
database["TARGET"] = [i for i in target["TARGET"]]
database.loc[database["TARGET"] == 1.0, "TARGET_STR"] = "Defaulted"
database.loc[database["TARGET"] == 0.0, "TARGET_STR"] = "Repayed"
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
    html.H4("Age groups compared to default percentage", style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(id='Age_group_graph'),

    html.H4("Analysis of EXT_SOURCE_1's effect", style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(id='EXT_SOURCE_1'),

    html.H4("Analysis of EXT_SOURCE_2's effect", style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(id='EXT_SOURCE_2'),

    html.H4("Analysis of CODE_GENDER's effect", style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(id='CODE_GENDER'),
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
def show_client_position_age_group_graph(client_id):
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
    fig.update_traces(marker_color='green')
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round(database.loc[client_id, "DAYS_BIRTH"] * 69.120548) % 10,
            # scaling back to full range from mapped value on [0,1]
            # % 10 because of the number of bins
            line_width=3, line_dash="dash",
            line_color="red")
        return fig
    return fig


@app.callback(
    Output("EXT_SOURCE_1", "figure"),
    Input("client_id", "value"))
def display_graph_EXT_SOURCE_1(client_id):
    fig = px.histogram(
        database, x="EXT_SOURCE_1",
        range_x=[0, 1],
        nbins=50,
        barmode="relative",
        color="TARGET_STR",
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        hover_data=database.columns)
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "EXT_SOURCE_1"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="red")
        return fig

        # fig.update_traces(marker_color='green')
    return fig


@app.callback(
    Output("EXT_SOURCE_2", "figure"),
    Input("client_id", "value"))
def display_graph_EXT_SOURCE_2(client_id):
    fig = px.histogram(
        database, x="EXT_SOURCE_2",
        range_x=[0, 1],
        nbins=50,
        barmode="relative",
        color="TARGET_STR",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        log_y=True,
        hover_data=database.columns)
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "EXT_SOURCE_2"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="red")
        return fig

        # fig.update_traces(marker_color='green')
    return fig


@app.callback(
    Output("CODE_GENDER", "figure"),
    Input("client_id", "value"))
def display_graph_CODE_GENDER(client_id):
    fig = fig = px.histogram(
        database,
        x="CODE_GENDER",
        color="TARGET_STR",
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "CODE_GENDER"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="red")
        return fig

        # fig.update_traces(marker_color='green')
    return fig


### Run the App ###############################################
if __name__ == '__main__':
    app.run_server(debug=False)
