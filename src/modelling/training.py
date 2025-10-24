from sklearn.ensemble import RandomForestRegressor
from .utils import pickle_object
from prefect import task
import os


@task
def train(
    X,
    y,
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    savepath=None,
):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    rf.fit(X, y)

    if savepath:
        pickle_object(rf, os.path.join(savepath, "model.pkl"))

    return rf
