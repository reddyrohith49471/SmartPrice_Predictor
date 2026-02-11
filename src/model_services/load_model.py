import mlflow
import mlflow.pyfunc
from src.config.config import Config

class ModelLoader:
    def __init__(self):
        self.config = Config()
        self.model = self._load_model()
    def _load_model(self):
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        model_uri = (
            f"models:/{self.config.MODEL_NAME}/{self.config.MODEL_STAGE}"
        )

        model = mlflow.pyfunc.load_model(model_uri)
        return model

    def get_model(self):
        return self.model
