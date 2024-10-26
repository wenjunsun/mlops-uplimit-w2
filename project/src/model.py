import numpy as np
import onnxruntime as rt
import wandb

from src.constants import WANDB_API_KEY, WANDB_OLD_MODEL_NAME, WANDB_NEW_MODEL_NAME


# NOTE: If your implementation uses a different model do update the methods
# load_model & predict accordingly!
class Model:
    @classmethod
    def load_model(cls, wandb_model_name) -> rt.InferenceSession:
        if WANDB_API_KEY is None:
            raise ValueError(
                "WANDB_API_KEY not set, unable to pull the model!",
            )
        run = wandb.init()
        downloaded_model_path = run.use_model(
            name=wandb_model_name,
        )
        return rt.InferenceSession(
            downloaded_model_path, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def load_old_model(cls) -> rt.InferenceSession:
        return cls.load_model(WANDB_OLD_MODEL_NAME)

    @classmethod
    def load_new_model(cls) -> rt.InferenceSession:
        return cls.load_model(WANDB_NEW_MODEL_NAME)

    @classmethod
    def predict(
        cls, session: rt.InferenceSession, review: str
    ) -> dict[
        int,
        float,
    ]:
        input_name = session.get_inputs()[0].name
        _, probas = session.run(None, {input_name: np.array([[review]])})
        return probas[0]
