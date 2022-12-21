
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
filepath = "/Users/loicvalenti/Library/Mobile Documents/com~apple~CloudDocs/Formation Data Science/PROJET 7/Project-7-Credit-Score/test_model_NN1 (1).csv"
client_predictions = pd.read_csv(filepath)
app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = request.form.values()
    prediction = client_predictions[client_predictions["SK_ID_CURR"] == int_features ]["TARGET"]

    output = round(prediction, 2)

    return render_template(
        "index.html", prediction_text="Predicted Salary $ {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)
