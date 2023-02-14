import pandas as pd
import uvicorn
from fastapi import FastAPI

filepath_predict_probs = "Credit Application Predict Probabilities.csv"

client_predictions = pd.read_csv(filepath_predict_probs)

app = FastAPI()


@app.get("/prediction/{client_id}")
async def root(client_id: int):
    client_id = int(client_id)
    if client_id in client_predictions["SK_ID_CURR"].values:

        # Insert the api request here
        prediction = client_predictions.loc[client_predictions["SK_ID_CURR"] == client_id].iloc[-1, 1]
        if prediction > 0.5:  # decision threshold customisable for risk level
            output = "Client's application was refused with {}% risk of defaulting".format(round(prediction * 100))
        else:
            output = "Client's application was accepted with {}% chance of servicing the debt".format(round(
                (1 - prediction) * 100))
    else:
        output = "Enter a valid client number in the space above"

    return {"message": output}


if __name__ == "__main__":
    uvicorn.run(app)
