
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import lime as lime
filepath = "Credit Application Results.csv"
client_predictions = pd.read_csv(filepath)
app = Flask(__name__)

"""
to unpickle the lime explainer and provide the feature importance explanation.
with open(explainer_filename, 'rb') as f: explainer = dill.load(f)
"""
"""
Utils
"""

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    client_id = [int(x) for x in request.form.values()][0]
    if client_id in client_predictions["SK_ID_CURR"].values:
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1,1]
        
        """
        Place holder for the graph printing methods and so on.
        """
        
        
        
        if prediction > 0.5000000:
            
            return render_template(
                    "index.html", prediction_text="Client's application was refused with {}% risk of defaulting".format(prediction*100)
                )
        else:
            
            return render_template(
                    "index.html", prediction_text="Client's application was accepted with {}% risk of defaulting".format(prediction*100)
                )
    else:
        return render_template(
                "index.html", prediction_text="Client's application is not registered in the database"
                    )
            

if __name__ == "__main__":
    app.run(debug=True)
