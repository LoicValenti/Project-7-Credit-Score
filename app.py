
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
def draw_age_graph():
    """
    Plots the age distribution with respect to the default probability and the applicant's position
    
    Parameters :
    Return :
    """
    
    age_data = app_train[['TARGET', 'DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

    # Bin the age data
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    # Group by the bin and calculate averages
    age_groups  = age_data.groupby('YEARS_BINNED').mean()
    plt.figure(figsize = (8, 8))

    # Graph the age bins and the average of the target as a bar plot
    plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

    # Plot labeling
    plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group');
    
    return
    
def draw_salary_graph():
    """
    Plots the salary distribution with respect to the default probability and the applicant's position
    
    Parameters :
    Return :
    """
    return
    
def draw_credit_graph():
    """
    Plots the credit amount distribution with respect to the default probability and the applicant's position
    
    Parameters :
    Return :
    """
    return`
    
def feature_importance_lime():
    """
    Plots the feature importance of the model for the applicant
    
    Parameters :
    Return :
    """
    return





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
