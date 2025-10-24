from pathlib import Path


class Paths:
    project_root = Path(__file__).resolve().parent.parent.parent
    path_data = project_root / "data" / "abalone_clean.csv"
    path_model = project_root / "src" / "web_service" / "local_objects" / "model.pkl"
