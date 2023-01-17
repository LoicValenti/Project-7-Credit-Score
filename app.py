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

"""
Initialization
"""

filepath = "Credit Application Results.csv"  # Prediction file, [Client_ID, Prediction probability]
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
    return (max_wanted - min_wanted) * (i - actual_max) / (actual_min - actual_max) + min_wanted


def do_initialization_of_databases():
    client_predictions = pd.read_csv(filepath)  # can be optimized, is there because of legacy reasons
    target_encoded = pd.read_csv(filepath)
    client_info_database = pd.read_csv(filepath_database)
    client_info_database = client_info_database.set_index(client_info_database["SK_ID_CURR"]).drop(
        columns=["SK_ID_CURR"])
    return client_predictions, target_encoded, client_info_database


client_predictions, target_encoded, client_info_database = do_initialization_of_databases()

client_info_database["DAYS_BIRTH"] = [rescaling(i, 20.09035, 68.98016, 0, 1) for i in
                                      client_info_database[
                                          "DAYS_BIRTH"]]  # scaling back from [0,1] to full range [20, 69]

client_info_database["DAYS_EMPLOYED"] = [rescaling(i, 0.002737851, 47.81109, 0, 1) for i in
                                         client_info_database[
                                             "DAYS_EMPLOYED"]]  # scaling back from [0,1] to full range [0, 49]

client_info_database["AMT_CREDIT"] = [rescaling(i, 4.500000e+04, 2.245500e+06, 0, 1) for i in
                                      client_info_database[
                                          "AMT_CREDIT"]]  # scaling back from [0,1] to full range [20, 69]

client_info_database["AMT_ANNUITY"] = [rescaling(i, 2295.000000, 180576.000000, 0, 1) for i in
                                       client_info_database[
                                           "AMT_ANNUITY"]]  # scaling back from [0,1] to full range [20, 69]

# Variable names

variable_indicators = ["Age group comparison", 'External source 1 comparison', "External source 2 comparison",
                       "Duration of employment comparison", "Age group detailed comparison", "Car ownership comparison",
                       "Credit amount comparison", "Credit annuity comparison", "Gender distribution comparison"]

age_groups = pd.read_csv(filepath_age_groups)
### load ML model ###########################################
### App Layout ###############################################
app.layout = html.Div(children=[
    html.H1(children='Interactive Client Application Reviewer', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-family': "Arial"
    }),
    dcc.Input(id='client_id', placeholder="Client's ID", value='Client ID', type="number", min=2, max=1000000, style={
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

### Callback to produce the prediction #########################
"""
@app.callback(
    Output("prediction_graph", "children"),
    Input('client_id', 'value')
)
def prediction_visualisation(client_id):
    return 
"""

"""
html.Div([
    dcc.Graph(id='graph_output')
]),
"""


@app.callback(
    Output('prediction_output', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1, 1]
        if prediction > 0.5000000:
            output = "Client's application was refused with {}% risk of defaulting".format(prediction * 100)
        else:
            output = "Client's application was accepted with {}% chance of servicing the debt".format(
                (1 - prediction) * 100)

    else:
        output = "Client's application is not in the database"

    return f'{output}.'


# Skeleton for the new graphing function
@app.callback(
    [Output(component_id='graph_output', component_property='figure'),
     Output('graph_output_explanation', 'children')],
    [Input("variable_choice", "value"),
     Input("client_id", "value")]
)
def trace_graph(variable_choice, client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        if variable_choice == 'External source 1 comparison':
            return display_graph_EXT_SOURCE_1(client_id), update_output_EXT_SOURCE_1(client_id)
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


def update_output_EXT_SOURCE_1(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "External Source 1 is a credit score rating from other banking agencies." \
                 " Client number: " + str(client_id) + " placed " \
                 + str(client_info_database.loc[client_id, "EXT_SOURCE_1"]) + \
                 " on this metric. \n" \
                 " The higher your score on this metric the better." \
                 " Client number {} placed on the {}th percentile." \
                 " It is {} away from the median of customers" \
                 " that serviced the debt obligations".format(
                     client_id,
                     round(stats.percentileofscore(
                         client_info_database["EXT_SOURCE_1"],
                         client_info_database.loc[client_id, "EXT_SOURCE_1"])),
                     round(abs(client_info_database.loc[
                                   client_info_database["TARGET_STR"] == "Repayed",
                                   "EXT_SOURCE_1"].median() -
                               client_info_database.loc[client_id, "EXT_SOURCE_1"]), 3))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


def update_output_EXT_SOURCE_2(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "External Source 2 is a credit score rating from other banking agencies." \
                 " Client number: " + str(client_id) + " placed " \
                 + str(client_info_database.loc[client_id, "EXT_SOURCE_2"]) + \
                 " on this metric." \
                 " The higher your score on this metric the better. Client number {} placed on the {}th percentile. " \
                 " Your score was {} away from the median of customers that serviced the debt obligations".format(
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
                 + str(client_info_database.loc[client_id, "DAYS_BIRTH"]) + \
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
                 + str(client_info_database.loc[client_id, "DAYS_EMPLOYED"]) + \
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
                 " dollars of credit. \n" \
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
                 " dollars of annuity. " + + \
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
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round(client_info_database.loc[client_id, "DAYS_BIRTH"]) % 10,
            line_width=3, line_dash="dash",
            line_color="cyan")
        return fig
    return fig


def display_graph_EXT_SOURCE_1(client_id):
    fig = px.histogram(
        client_info_database, x="EXT_SOURCE_1",
        range_x=[0, 1],
        barmode="relative",
        marginal="box",
        color="TARGET_STR",
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
        hover_data=client_info_database.columns,
        title="External credit rating 1 for client %s" % client_id,
        labels={
            "x": "External Credit Rating 1",
            "y": "Count"
        })
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((client_info_database.loc[client_id, "EXT_SOURCE_1"]) * 100) / 100,
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


### Run the App ###############################################
if __name__ == '__main__':
    app.run_server(debug=True)
