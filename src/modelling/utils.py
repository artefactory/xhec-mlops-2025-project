import pickle
from pathlib import Path


def pickle_object(obj, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {output_path}")


def load_pickle(input_path: Path):
    with open(input_path, "rb") as f:
        return pickle.load(f)
