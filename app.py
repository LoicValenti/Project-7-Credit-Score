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
import scipy.stats as stats

# import pickle
### Setup ###################################################
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
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
database["DAYS_BIRTH"] *= 69
database["DAYS_BIRTH"] += 20
database["DAYS_EMPLOYED"] *= 17912.000000 / 365
database["AMT_CREDIT"] *= 4.050000e+06
database["AMT_ANNUITY"] *= 258025.500000

# Variable names

variable_indicators = ["Age_group", 'EXT_SOURCE_1', "EXT_SOURCE_2", "DAYS_EMPLOYED", "DAYS_BIRTH", "FLAG_OWN_CAR",
                       "AMT_CREDIT", "AMT_ANNUITY", " CODE_GENDER"]

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
    dcc.Dropdown(
        id="variable_choice",
        options=[{"label": i, "value": i} for i in variable_indicators],
        placeholder="Select graph"
    ),
    html.Div(dcc.Graph(id='graph_output')),
    """
        html.H4("Age groups compared to default percentage", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(id='Age_group_graph'),
    
        html.H4("Analysis of the first external credit rating", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='EXT_SOURCE_1'),
    
        html.Div(id='explanation_EXT_SOURCE_1', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Analysis of second external credit rating", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='EXT_SOURCE_2'),
    
        html.Div(id='explanation_EXT_SOURCE_2', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Analysis of the gender's default behaviour", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='CODE_GENDER'),
    
        html.Div(id='explanation_CODE_GENDER', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Analysis of car owners default behaviour ", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='FLAG_OWN_CAR'),
    
        html.Div(id='explanation_FLAG_OWN_CAR', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Analysis of the effect of age", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='DAYS_BIRTH'),
    
        html.Div(id='explanation_DAYS_BIRTH', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Analysis of the number of years of employment", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='DAYS_EMPLOYED'),
    
        html.Div(id='explanation_DAYS_EMPLOYED', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Description of default compared to the amount proposed", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='AMT_CREDIT'),
    
        html.Div(id='explanation_AMT_CREDIT', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    
        html.H4("Description of default compared to the annuity ", style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='AMT_ANNUITY'),
    
        html.Div(id='explanation_AMT_ANNUITY', style={
            'textAlign': 'left',
            'color': colors['text']
        }),
    """
])


### Callback to produce the prediction #########################

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
    Output('graph_output', 'children'),
    [Input("variable_choice", "value"),
     Input("client_id", "value")]
)
def trace_graph(variable_choice, client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        if variable_choice == 'EXT_SOURCE_1':
            fig = px.histogram(
                database, x="EXT_SOURCE_1",
                range_x=[0, 1],
                nbins=50,
                barmode="relative",
                marginal="box",
                color="TARGET_STR",
                log_y=True,
                color_discrete_sequence=px.colors.qualitative.Alphabet_r,
                hover_data=database.columns)
            fig.update_layout(
                bargap=0.01,
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'])
            fig.add_vline(
                x=round((database.loc[client_id, "EXT_SOURCE_1"]) * 100) / 100,
                line_width=3, line_dash="dash",
                line_color="blue")
            return fig
    return "Client is not in the database"


@app.callback(
    Output('explanation_EXT_SOURCE_1', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "External Source 1 is a credit score rating from other banking agencies." \
                 "The higher your score on this metric the better. Client number {} placed on the {}th percentile. " \
                 "Your score was {} away from the median of customers that serviced the debt obligations".format(
            client_id,
            round(stats.percentileofscore(
                database["EXT_SOURCE_1"],
                database.loc[client_id, "EXT_SOURCE_1"])),
            abs(database.loc[
                    database["TARGET_STR"] == "Repayed",
                    "EXT_SOURCE_1"].median() -
                database.loc[client_id, "EXT_SOURCE_1"]))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_EXT_SOURCE_2', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "External Source 2 is a credit score rating from other banking agencies." \
                 "The higher your score on this metric the better. Client number {} placed on the {}th percentile. " \
                 "Your score was {} away from the median of customers that serviced the debt obligations".format(
            client_id,
            round(stats.percentileofscore(
                database["EXT_SOURCE_2"],
                database.loc[client_id, "EXT_SOURCE_2"])),
            abs(database.loc[
                    database["TARGET_STR"] == "Repayed",
                    "EXT_SOURCE_2"].median() -
                database.loc[client_id, "EXT_SOURCE_2"]))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_CODE_GENDER', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "People amongst the gender group 0 have a much higher risk of defaulting than the gender group 1"
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_FLAG_OWN_CAR', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1, 1]
        if prediction > 0.5000000:
            output = "Placeholder for explanation of this variable's effect on the refusal"
        else:
            output = "Placeholder for explanation of this variable's effect on the acceptance"

    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_DAYS_BIRTH', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "The client's age is a strong factor for prediction of default." \
                 "Client number {} placed on the {}th percentile. " \
                 "The client is {} away from the median of customers that serviced the debt obligations".format(
            client_id,
            round(stats.percentileofscore(
                database["DAYS_BIRTH"],
                database.loc[client_id, "DAYS_BIRTH"])),
            abs(database.loc[
                    database["TARGET_STR"] == "Repayed",
                    "DAYS_BIRTH"].median() -
                database.loc[client_id, "DAYS_BIRTH"]))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_DAYS_EMPLOYED', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "The client's number of years of employment is a strong factor for prediction of default." \
                 "Client number {} placed on the {}th percentile. " \
                 "The client is {} away from the median of customers that serviced the debt obligations".format(
            client_id,
            round(stats.percentileofscore(
                database["DAYS_EMPLOYED"],
                database.loc[client_id, "DAYS_EMPLOYED"])),
            abs(database.loc[
                    database["TARGET_STR"] == "Repayed",
                    "DAYS_EMPLOYED"].median() -
                database.loc[client_id, "DAYS_EMPLOYED"]))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_AMT_CREDIT', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "The amount of the credit asked by the customer." \
                 "Client number {} placed on the {}th percentile. " \
                 "The client asked for a credit amount {} away from the median of" \
                 " customers that serviced the debt obligations".format(
            client_id,
            round(stats.percentileofscore(
                database["AMT_CREDIT"],
                database.loc[client_id, "AMT_CREDIT"])),
            abs(database.loc[
                    database["TARGET_STR"] == "Repayed",
                    "AMT_CREDIT"].median() -
                database.loc[client_id, "AMT_CREDIT"]))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


@app.callback(
    Output('explanation_AMT_ANNUITY', 'children'),
    Input('client_id', 'value'))
def update_output(client_id):
    if client_id in client_predictions["SK_ID_CURR"].values:
        output = "The amount of the annuity." \
                 "Client number {} placed on the {}th percentile. " \
                 "The client asked for a credit amount {} away from the median of" \
                 " customers that serviced the debt obligations".format(
            client_id,
            round(stats.percentileofscore(
                database["AMT_ANNUITY"],
                database.loc[client_id, "AMT_ANNUITY"])),
            abs(database.loc[
                    database["TARGET_STR"] == "Repayed",
                    "AMT_ANNUITY"].median() -
                database.loc[client_id, "AMT_ANNUITY"]))
    else:
        output = "Client's application is not in the database"

    return f'{output}.'


"""

@app.callback(
    Output("dictionnary", 'children'),
    Input('prediction_output', 'children'))
def update_explanation(client_id):
    dictionary = dict()
    if client_id in client_predictions["SK_ID_CURR"].values:
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1, 1]

        if prediction > 0.5000000:
            dictionary["explanation_EXT_SOURCE_1"] = "placeholder"
            dictionary["explanation_EXT_SOURCE_2"] = "placeholder"
            dictionary["explanation_CODE_GENDER"] = "placeholder"
            dictionary["explanation_FLAG_OWN_CAR"] = "placeholder"
            dictionary["explanation_DAYS_BIRTH"] = "placeholder"
            dictionary["explanation_DAYS_EMPLOYED"] = "placeholder"
            dictionary["explanation_AMT_CREDIT"] = "placeholder"
            dictionary["explanation_AMT_ANNUITY"] = "placeholder"
        else:
            dictionary["explanation_EXT_SOURCE_1"] = "placeholder"
            dictionary["explanation_EXT_SOURCE_2"] = "placeholder"
            dictionary["explanation_CODE_GENDER"] = "placeholder"
            dictionary["explanation_FLAG_OWN_CAR"] = "placeholder"
            dictionary["explanation_DAYS_BIRTH"] = "placeholder"
            dictionary["explanation_DAYS_EMPLOYED"] = "placeholder"
            dictionary["explanation_AMT_CREDIT"] = "placeholder"
            dictionary["explanation_AMT_ANNUITY"] = "placeholder"

    else:
        dictionary["not_in_database"] = "placeholder"

    return dictionary
"""


@app.callback(
    Output('Age_group_graph', "children"),
    Input('client_id', 'value'))
def show_client_position_age_group_graph(client_id):
    x_ticks_labels = ["(20.0, 25.0", "(25.0, 30.0]", "(30.0, 35.0]", "(35.0, 40.0]", "(40.0, 45.0]", "(45.0, 50.0]",
                      "(50.0, 55.0]", "(55.0, 60.0]", "(60.0, 65.0]", "(65.0, 70.0]"]

    fig = px.bar(x=x_ticks_labels, y=100 * age_groups['TARGET'],
                 color_discrete_sequence=px.colors.qualitative.Alphabet_r,
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
            x=round(database.loc[client_id, "DAYS_BIRTH"]) % 10,
            line_width=3, line_dash="dash",
            line_color="blue")
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
        marginal="box",
        color="TARGET_STR",
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
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
            line_color="blue")
        return fig

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
        marginal="box",
        color="TARGET_STR",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
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
            line_color="blue")
        return fig

    return fig


@app.callback(
    Output("CODE_GENDER", "figure"),
    Input("client_id", "value"))
def display_graph_CODE_GENDER(client_id):
    fig = px.histogram(
        database,
        x="CODE_GENDER",
        color="TARGET_STR",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
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
            line_color="blue")
        return fig

    return fig


@app.callback(
    Output("FLAG_OWN_CAR", "figure"),
    Input("client_id", "value"))
def display_graph_FLAG_OWN_CAR(client_id):
    fig = px.histogram(
        database,
        x="FLAG_OWN_CAR",
        color="TARGET_STR",
        barmode="group",
        histnorm="percent",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,

    )
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "FLAG_OWN_CAR"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


@app.callback(
    Output("DAYS_BIRTH", "figure"),
    Input("client_id", "value"))
def display_graph_DAYS_BIRTH(client_id):
    fig = px.histogram(
        database,
        x="DAYS_BIRTH",
        color="TARGET_STR",
        marginal="box",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
    )
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "DAYS_BIRTH"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


@app.callback(
    Output("DAYS_EMPLOYED", "figure"),
    Input("client_id", "value"))
def display_graph_DAYS_EMPLOYED(client_id):
    fig = px.histogram(
        database,
        x="DAYS_EMPLOYED",
        color="TARGET_STR",
        barmode="group",
        marginal="box",
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,

    )
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "DAYS_EMPLOYED"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


@app.callback(
    Output("AMT_CREDIT", "figure"),
    Input("client_id", "value"))
def display_graph_AMT_CREDIT(client_id):
    fig = px.histogram(
        database,
        x="AMT_CREDIT",
        color="TARGET_STR",
        log_y=True,
        marginal="box",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
    )
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "AMT_CREDIT"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


@app.callback(
    Output("AMT_ANNUITY", "figure"),
    Input("client_id", "value"))
def display_graph_AMT_ANNUITY(client_id):
    fig = px.histogram(
        database,
        x="AMT_ANNUITY",
        color="TARGET_STR",
        marginal="box",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Alphabet_r,
    )
    fig.update_layout(
        bargap=0.01,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
    if client_id in client_predictions["SK_ID_CURR"].values:
        fig.add_vline(
            x=round((database.loc[client_id, "AMT_ANNUITY"]) * 100) / 100,
            line_width=3, line_dash="dash",
            line_color="blue")
        return fig

    return fig


### Run the App ###############################################
if __name__ == '__main__':
    app.run_server(debug=True)
