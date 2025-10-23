from pydantic import BaseModel
from typing import List

""" not needed / used, but leave it ?
class PredictionInput(BaseModel):
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float
    rings: int
    sex: str

class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]
"""


class BatchPredictionOutput(BaseModel):
    predictions: List
