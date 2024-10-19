import os
from enum import Enum


class SentimentLabel(str, Enum):
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


LABEL_CLASS_TO_NAME = {
    0: SentimentLabel.NEGATIVE.value,
    1: SentimentLabel.NEUTRAL.value,
    2: SentimentLabel.POSITIVE.value,
}


WANDB_MODEL_REGISTRY_MODEL_NAME = "hughsun/model-registry/review-sentiment-analysis-dev:v3"

# NOTE: Ensure that you set the API Key within Github Codespaces secrets
# in the settings page of your repository!
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
