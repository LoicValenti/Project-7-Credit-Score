"""
First step : Imports
"""

import pandas as pd  # Data processing
import plotly.express as px  # Visualization
import dash  # Dashboard
from dash import dcc  # Dashboard
from dash import html  # Dashboard
from dash.dependencies import Input, Output, State  # Callback functions for the dashboard
import scipy.stats as stats  # Stats module
import requests  # Calling the API
import json  # formating api get in json

"""
Initialization
"""

filepath = "Credit Application Results.csv"  # Prediction file, [Client_ID, Prediction probability]
filepath_predict_probs = "Credit Application Predict Probabilities.csv"
filepath_database = "Dataset_for_webapp.csv"  # Data file for the computations
filepath_age_groups = "age_groups.csv"  # Data file for the age groups graph, could be optimized

# Initialize the server app

app = dash.Dash(__name__)
app.title = 'Machine Learning Model Deployment'
server = app.server

# Initialize the dashboard's colors

colors = {
    'background': '#000000',
    'text': '#FFFBF6'
}


# Some utility functions
def rescaling(i, min_wanted, max_wanted, actual_min, actual_max):
    """
    Rescales the data from the given min and max to the given values
    To be used in a list creation function

    """
    return (max_wanted - min_wanted) * (i - actual_max) / (actual_min - actual_max) + min_wanted


def do_initialization_of_databases():
    """
    Initialize the databases used by the application.

    Parameters :
    None
    Returns :
    client_predictions : pandas dataframe containing the prediction probabilities
    target_encoded : pandas dataframe containing the encoded target to 0 or 1
    client_info_database : pandas dataframe containing the client information

    """
    client_predictions = pd.read_csv(filepath_predict_probs)  # can be optimized, is there because of legacy reasons
    target_encoded = pd.read_csv(filepath)
    client_info_database = pd.read_csv(filepath_database)
    client_info_database = client_info_database.set_index(client_info_database["SK_ID_CURR"]).drop(
        columns=["SK_ID_CURR"])
    return client_predictions, target_encoded, client_info_database


client_predictions, target_encoded, client_info_database = do_initialization_of_databases()

# Rescaling of the database from [0,1] from the training to the actual min and max

client_info_database["DAYS_BIRTH"] = [rescaling(i, 20.09035, 68.98016, 0, 1) for i in
                                      client_info_database[
                                          "DAYS_BIRTH"]]  # scaling back from [0,1] to full range [20, 69]

client_info_database["DAYS_EMPLOYED"] = [rescaling(i, 0.002737851, 47.81109, 0, 1) for i in
                                         client_info_database[
                                             "DAYS_EMPLOYED"]]  # scaling back from [0,1] to full range [0, 47]

client_info_database["AMT_CREDIT"] = [rescaling(i, 4.500000e+04, 2.245500e+06, 0, 1) for i in
                                      client_info_database[
                                          "AMT_CREDIT"]]  # scaling back from [0,1]
# to full range [4.500000e+04, 2.245500e+06]

client_info_database["AMT_ANNUITY"] = [rescaling(i, 2295.0, 180576.0, 0, 1) for i in
                                       client_info_database[
                                           "AMT_ANNUITY"]]  # scaling back from [0,1] to full range [2295, 180576]

# Variable names for the dropdown list

variable_indicators = ["Age group comparison", 'External source 3 comparison', "External source 2 comparison",
                       "Duration of employment comparison", "Age group detailed comparison", "Car ownership comparison",
                       "Credit amount comparison", "Credit annuity comparison", "Gender distribution comparison"]

# age_groups is a special case dataframe to plot the first graphs that appears on the dashboard as a bandaid fix
# for a bug in printing the graph correctly

age_groups = pd.read_csv(filepath_age_groups)

"""
HTML layout of the dashboard
"""

app.layout = html.Div(children=[
    html.H1(children='Interactive Client Application Reviewer', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-family': "Arial"
    }),
    dcc.Input(id='client_id', placeholder="Client's ID", value='1', type="number", min=1, max=1000000,
              style={
                  'textAlign': 'left',
                  'color': colors['background'],
                  'font-family': "Arial",
                  'width': '22.5%',
                  'height': '27px',
                  'display': 'inline-block',
                  'margin-bottom': '15px'
              }),
    html.Div(id='prediction_output', style={
        'textAlign': 'left',
        'color': colors['text'],
        'font-family': "Arial",
        'font-size': "20px",
        'margin-bottom': '15px'
    }),
    html.Div(id='prediction_output_personal_information', style={
        'textAlign': 'left',
        'color': colors['text'],
        'font-family': "Arial",
        'font-size': "20px",
        'margin-bottom': '15px'
    }),
    dcc.Dropdown(
        id="variable_choice",
        options=[{"label": i, "value": i} for i in variable_indicators],
        placeholder="Select graph",
        style={'width': '48%',
               'display': 'inline-block',
               'font-family': "Arial"
               }
    ),
    html.Div([
        dcc.Graph(id='graph_output')
    ], style={
        'margin-bottom': '15px'
    }),
    html.Div(
        [
            html.Div(
                [
                    html.Div(id='graph_output_explanation', style={
                        'color': colors['text'],
                        'font-family': "Arial",
                        'font-size': '20px',
                        'textAlign': 'center',
                        'width': "100%"

                    }
                             )
                ],
                style={
                    "width": '75%',
                    "margin": "0 auto"
                },
            )
        ], style={'width': '100%', })
]
)

"""
url = "https://random-facts2.p.rapidapi.com/getfact"
headers = {
    'x-rapidapi-host': "random-facts2.p.rapidapi.com",
    'x-rapidapi-key': "YOUR-RAPIDAPI-HUB-Key"
}
response = requests.request("GET", url, headers=headers)
print(response.text)
"""


# Callback to produce the prediction #########################

@app.callback(
    Output('prediction_output', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    """
    Update the output for the client's prediction

    Parameters:
        Client_id (int): Client identifier in the database

    Returns:
        output (string): Model prediction for the client
    """
    """
    if client_id in client_predictions["SK_ID_CURR"].values:

        # Insert the api request here
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1, 1]
        if prediction > 0.5000000:  # Legacy function needs to be updated to reduce memory usage
            output = "Client's application was refused with {}% risk of defaulting".format(round(prediction * 100))
        else:
            output = "Client's application was accepted with {}% chance of servicing the debt".format(round(
                (1 - prediction) * 100))

    else:
        output = "Client's application is not in the database"
        
    """
    url = "http://127.0.0.1:8000/prediction/{}".format(client_id)
    output = requests.request("GET", url)
    return f'{json.loads(output.text)["message"]}.'


@app.callback(
    Output('prediction_output_personal_information', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    """
    Update the output for the client's profile

    Parameters:
        Client_id (int): Client identifier in the database

    Returns:
        output (string): Profile sentence for the client
    """
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "Client number " + str(client_id) + \
                 " is " + str(round(client_info_database.loc[client_id, "DAYS_BIRTH"])) + \
                 " years old, has " + str(round(client_info_database.loc[client_id, "CNT_CHILDREN"])) + \
                 " children, has been employed " + str(round(client_info_database.loc[client_id, "DAYS_EMPLOYED"])) + \
                 " years, and earns " + str(
            round(client_info_database.loc[client_id, "AMT_INCOME_TOTAL"]))

        return f'{output}.'
    return ''


# Skeleton for the new graphing function
@app.callback(
    [Output(component_id='graph_output', component_property='figure'),
     Output('graph_output_explanation', 'children')],
    [Input("variable_choice", "value"),
     Input("client_id", "value")]
)
def trace_graph(variable_choice, client_id):
    """
    Main graph function.

    Gives the html the appropriate graph from the dropdown list.
    Also feeds the corresponding explanation.

    Parameters:
        client_id (int): Client identifier in the database
        variable_choice (str): graph chose from the dropdown list

    Returns:
        graph_output (px object): graph output from the dropdown list
        graph_output_explanation (string): graph output explanation
    """

    if client_id in client_predictions["SK_ID_CURR"].values:  # No need to go through all the if statements as
        # they contain a return statement. It could be improved with a refactoring of the function's name
        if variable_choice == 'External source 3 comparison':
            return display_graph_EXT_SOURCE_3(client_id), update_output_EXT_SOURCE_3(client_id)
        if variable_choice == 'External source 2 comparison':
            return display_graph_EXT_SOURCE_2(client_id), update_output_EXT_SOURCE_2(client_id)
        if variable_choice == 'Gender distribution comparison':
            return display_graph_CODE_GENDER(client_id), update_output_CODE_GENDER(client_id)
        if variable_choice == 'Car ownership comparison':
            return display_graph_FLAG_OWN_CAR(client_id), update_output_FLAG_OWN_CAR(client_id)
        if variable_choice == 'Age group detailed comparison':
            return display_graph_DAYS_BIRTH(client_id), update_output_DAYS_BIRTH(client_id)
        if variable_choice == 'Duration of employment comparison':
            return display_graph_DAYS_EMPLOYED(client_id), update_output_DAYS_EMPLOYED(client_id)
        if variable_choice == 'Credit amount comparison':
            return display_graph_AMT_CREDIT(client_id), update_output_AMT_CREDIT(client_id)
        if variable_choice == 'Credit annuity comparison':
            return display_graph_AMT_ANNUITY(client_id), update_output_AMT_ANNUITY(client_id)
        if variable_choice == "Age group comparison":
            return show_client_position_age_group_graph(client_id), "Age group comparison"

    x_ticks_labels = ["(20.0, 25.0", "(25.0, 30.0]", "(30.0, 35.0]", "(35.0, 40.0]", "(40.0, 45.0]", "(45.0, 50.0]",
                      "(50.0, 55.0]", "(55.0, 60.0]", "(60.0, 65.0]", "(65.0, 70.0]"]

    fig = px.bar(x=x_ticks_labels, y=100 * age_groups['TARGET'],
                 color_discrete_sequence=px.colors.qualitative.Alphabet_r,
                 title="Failure to Repay by Age Group",
                 labels={
                     "x": "Age groups",
                     "y": "Failure to Repay (%)"
                 })

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig, "Enter the necessary information above"


# all the other functions are utility functions to graph the appropriate figure for the main "trace_graph" function
# plotly.express is used to create the reactive graphs and are included in the dash ecosystem.
def update_output_EXT_SOURCE_3(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "External Source 3 is a credit score rating from other banking agencies." \
                 " Client number: " + str(client_id) + " placed " \
                 + str(round(client_info_database.loc[client_id, "EXT_SOURCE_3"], 3)) + \
                 " on this metric. \n" \
                 " The higher the score on this metric the better." \
                 " Client number {} placed on the {}th percentile." \
                 " It is {} away from the median of customers" \
                 " that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["EXT_SOURCE_3"],
                         client_info_database.loc[client_id, "EXT_SOURCE_3"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "EXT_SOURCE_3"].median() -
                               client_info_database.loc[client_id, "EXT_SOURCE_3"]), 3))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_EXT_SOURCE_2(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "External Source 2 is a credit score rating from other banking agencies." \
                 " Client number: " + str(client_id) + " placed " \
                 + str(round(client_info_database.loc[client_id, "EXT_SOURCE_2"], 3)) + \
                 " on this metric." \
                 " The higher the score on this metric the better. Client number {} placed on the {}th percentile. " \
                 " It is {} away from the median of customers that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["EXT_SOURCE_2"],
                         client_info_database.loc[client_id, "EXT_SOURCE_2"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "EXT_SOURCE_2"].median() -
                               client_info_database.loc[client_id, "EXT_SOURCE_2"]), 2))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_CODE_GENDER(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = " Client number: " + str(client_id) + " is part of the group " \
                 + str(client_info_database.loc[client_id, "CODE_GENDER_STR"]) + \
                 ". People amongst the gender group male have a much higher risk of " \
                 "defaulting than the gender group female"
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_FLAG_OWN_CAR(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = " Client number: " + str(client_id) + " is part of the group " \
                 + str(client_info_database.loc[client_id, "FLAG_OWN_CAR"]) + \
                 " People amongst the group 0 have higher risk of defaulting than the group 1"
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_DAYS_BIRTH(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "The client's age is a strong factor for prediction of default." \
                 " Client number: " + str(client_id) + " is " \
                 + str(round(client_info_database.loc[client_id, "DAYS_BIRTH"])) + \
                 " years old." \
                 " Client number {} placed on the {}th percentile. " \
                 " The client is {} away from the median of customers that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["DAYS_BIRTH"],
                         client_info_database.loc[client_id, "DAYS_BIRTH"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "DAYS_BIRTH"].median() -
                               client_info_database.loc[client_id, "DAYS_BIRTH"]), 2))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_DAYS_EMPLOYED(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "The client's number of years of employment is a strong factor for prediction of default." \
                 " Client number: " + str(client_id) + " has " \
                 + str(round(client_info_database.loc[client_id, "DAYS_EMPLOYED"])) + \
                 " years of experience." \
                 " Client number {} placed on the {}th percentile. " \
                 " The client is {} away from the median of customers that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["DAYS_EMPLOYED"],
                         client_info_database.loc[client_id, "DAYS_EMPLOYED"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "DAYS_EMPLOYED"].median() -
                               client_info_database.loc[client_id, "DAYS_EMPLOYED"]), 2))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_AMT_CREDIT(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "Client number: " + str(client_id) + " asked for " \
                 + str(round(client_info_database.loc[client_id, "AMT_CREDIT"])) + \
                 " of credit. \n" \
                 " Client number {} placed on the {}th percentile. " \
                 " The client asked for a credit amount {} away from the median of " \
                 " customers that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["AMT_CREDIT"],
                         client_info_database.loc[client_id, "AMT_CREDIT"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "AMT_CREDIT"].median() -
                               client_info_database.loc[client_id, "AMT_CREDIT"]), 2))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_AMT_ANNUITY(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "Client number: " + str(client_id) + " would have " \
                 + str(round(client_info_database.loc[client_id, "AMT_ANNUITY"])) + \
                 " of annuity. " + \
                 " Client number {} placed on the {}th percentile. " \
                 " The client's annuity are {} away from the median of" \
                 " customers that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["AMT_ANNUITY"],
                         client_info_database.loc[client_id, "AMT_ANNUITY"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "AMT_ANNUITY"].median() -
                               client_info_database.loc[client_id, "AMT_ANNUITY"]), 2))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def show_client_position_age_group_graph(client_id):
    x_ticks_labels = ["(20.0, 25.0", "(25.0, 30.0]", "(30.0, 35.0]", "(35.0, 40.0]", "(40.0, 45.0]", "(45.0, 50.0]",
                      "(50.0, 55.0]", "(55.0, 60.0]", "(60.0, 65.0]", "(65.0, 70.0]"]

    fig = px.bar(x=x_ticks_labels, y=100 * age_groups['TARGET'],
                 color_discrete_sequence=px.colors.qualitative.Alphabet_r,
                 title="Failure to Repay by Age Group",
                 labels={
                     "x": "Age Groups",
                     "y": "Failure to Repay (%)"
                 })

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    """
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round(client_info_database.loc[client_id, "DAYS_BIRTH"] / 10) + 1,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig
    """
    return fig


def display_graph_EXT_SOURCE_3(client_id):
    fig = px.histogram(
        client_info_database, x="EXT_SOURCE_3",
        range_x=[0, 1],
        barmode="relative",
        marginal="box",
        color="TARGET_STR",
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        hover_data=client_info_database.columns,
        title="External credit rating 3 for client %s" % client_id,
        labels={
            "x": "External Credit Rating 3",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "EXT_SOURCE_3"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig

    return fig


def display_graph_EXT_SOURCE_2(client_id):
    fig = px.histogram(
        client_info_database, x="EXT_SOURCE_2",
        range_x=[0, 1],
        nbins=50,
        barmode="relative",
        marginal="box",
        color="TARGET_STR",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        log_y=True,
        hover_data=client_info_database.columns,
        title="External credit rating 2 for client %s" % client_id,
        labels={
            "x": "External Credit Rating 2",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "EXT_SOURCE_2"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig

    return fig


def display_graph_CODE_GENDER(client_id):
    fig = px.histogram(
        client_info_database,
        x="CODE_GENDER_STR",
        color="TARGET_STR",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        title="Gender groups on default risk ",
        labels={
            "x": "Gender Groups",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "CODE_GENDER"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


def display_graph_FLAG_OWN_CAR(client_id):
    fig = px.histogram(
        client_info_database,
        x="FLAG_OWN_CAR",
        color="TARGET_STR",
        barmode="group",
        histnorm="percent",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        title="Car ownership",
        labels={
            "x": "Car Ownership Groups",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "FLAG_OWN_CAR"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


def display_graph_DAYS_BIRTH(client_id):
    fig = px.histogram(
        client_info_database,
        x="DAYS_BIRTH",
        color="TARGET_STR",
        marginal="box",
        barmode="relative",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        title="Age distribution for client %s" % client_id,
        labels={
            "x": "Age in Years",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=client_info_database.loc[client_id, "DAYS_BIRTH"],
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig

    return fig


def display_graph_DAYS_EMPLOYED(client_id):
    fig = px.histogram(
        client_info_database,
        x="DAYS_EMPLOYED",
        color="TARGET_STR",
        barmode="relative",
        marginal="box",
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        title="Duration of employment for client %s" % client_id,
        labels={
            "x": "Employment in Years",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "DAYS_EMPLOYED"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig

    return fig


def display_graph_AMT_CREDIT(client_id):
    fig = px.histogram(
        client_info_database,
        x="AMT_CREDIT",
        color="TARGET_STR",
        log_y=True,
        marginal="box",
        barmode="relative",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        title="Credit distribution for client %s" % client_id,
        labels={
            "x": "Credit Amount",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "AMT_CREDIT"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig

    return fig


def display_graph_AMT_ANNUITY(client_id):
    fig = px.histogram(
        client_info_database,
        x="AMT_ANNUITY",
        color="TARGET_STR",
        marginal="box",
        barmode="relative",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        title="Annuity distribution for client %s" % client_id,
        labels={
            "x": "Annuity Amount",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "AMT_ANNUITY"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig

    return fig


# Run the App ###############################################
if __name__ == '__main__':
    app.run_server(debug=True)
