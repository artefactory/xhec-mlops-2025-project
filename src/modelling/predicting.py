import pickle as pkl


def predict(X, model=None, savepath=None):
    if model:
        return model.predict(X)
    elif savepath:
        model = pkl.load(savepath)
        return model.predict(X)
    else:
        raise ValueError("Please specify a model to make a prediciton")
