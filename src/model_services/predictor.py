import pandas as pd
import numpy as np
from src.config.config import Config

class Predictor:
    def __init__(self, model):
        self.model = model
        self.config = Config()
    def predict(self, text: str, value: float):
        df = pd.DataFrame(
            [{"combined_text": text, "Value": value}]
        )
        pred_log = self.model.predict(df)
        price_clipped_pred = np.expm1(pred_log)
        price_pred = np.clip(price_clipped_pred, None, self.config.UPPER_LIMIT)
        return float(price_pred[0])
