from fastapi import FastAPI


from src.web_service.lib.models import BatchPredictionOutput
from src.modelling.predicting import predict_on_unseen_data
from src.web_service.utils import Paths


app = FastAPI(
    title="Making predictions on new data",
    description="this allows to use our ML model on unseen data and predict it.",
)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict_all", response_model=BatchPredictionOutput, status_code=201)
def predict() -> dict:
    """Make a prediction based on the pre loaded data."""
    prediction = predict_on_unseen_data(
        input__data_filepath=Paths.path_data, model_path=Paths.path_model
    )
    # convert NumPy array to list of dicts
    predictions = prediction.tolist()

    return {"predictions": predictions}
