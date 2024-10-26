import random
from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle

from src.data_models import SimpleModelRequest, SimpleModelResponse, SimpleModelResults
from src.model import Model

app = FastAPI(
    title="Drug Review Sentiment Analysis",
    description="Drug Review Sentiment Classifier",
    version="0.1",
)

# TODO: Add in appropriate logging using loguru wherever you see fit
# in order to aid with debugging issues.


@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
@serve.ingress(app)
class Canary:
    def __init__(self, old_model: DeploymentHandle, new_model: DeploymentHandle, canary_percent: float) -> None:
        self.old_model = old_model
        self.new_model = new_model
        self.canary_percent = canary_percent

    @app.post("/predict")
    async def predict(self, request: SimpleModelRequest) -> SimpleModelResponse:
        review = request.review
        if random.random() > self.canary_percent:
            result = await self.old_model.predict.remote(review)
        else:
            result = await self.new_model.predict.remote(review)
        return SimpleModelResponse.model_validate(result.model_dump())


@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class OldModel:
    def __init__(self) -> None:
        self.session = Model.load_old_model()

    def predict(self, review: str) -> SimpleModelResults:
        # In real life I would embed the model version in the API response itself or log it using a logger,
        # but in this simple project a simple print statement would do.
        print("Using old model!")
        result = Model.predict(self.session, review)
        return SimpleModelResults.model_validate(result)


@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class NewModel:
    def __init__(self) -> None:
        self.session = Model.load_new_model()

    def predict(self, review: str) -> SimpleModelResults:
        # In real life I would embed the model version in the API response itself or log it using a logger,
        # but in this simple project a simple print statement would do.
        print("Using new model!")
        result = Model.predict(self.session, review)
        return SimpleModelResults.model_validate(result)


entrypoint = Canary.bind(
    OldModel.bind(),
    NewModel.bind(),
    canary_percent = 0.2
)
