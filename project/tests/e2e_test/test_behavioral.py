import pytest
import requests

from src.constants import SentimentLabel
from src.data_models import SimpleModelRequest

@pytest.mark.parametrize(
    "input_a, input_b, label",
    [
        (
            "Hello world this is a good product!",
            "Hello world that is a good product!",
            SentimentLabel.POSITIVE.value,
        ),
    ],
)
def test_invariance(predict_url, input_a, input_b, label):
    """INVariance via verb injection (changes should not affect outputs)."""
    label_a = requests.post(predict_url, json=SimpleModelRequest(review=input_a).model_dump()).json()["label"]
    label_b = requests.post(predict_url, json=SimpleModelRequest(review=input_b).model_dump()).json()["label"]

    assert label_a == label_b == label


def test_directional(predict_url):
    """DIRectional expectations (changes with known outputs)."""
    good_review = SimpleModelRequest(review="Hello world this is a good product!").model_dump()
    really_good_review = SimpleModelRequest(review="Hello world this is really the best product ever!").model_dump()

    good_review_response = requests.post(predict_url, json=good_review).json()
    really_good_review_response = requests.post(predict_url, json=really_good_review).json()

    assert good_review_response['label'] == SentimentLabel.POSITIVE.value
    assert really_good_review_response['label'] == SentimentLabel.POSITIVE.value

    # model should give more probability on the really good review on positive emotion than a good review.
    assert really_good_review_response['score'] > good_review_response['score']


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "Hello world this is the best product ever!",
            SentimentLabel.POSITIVE.value,
        ),
    ],
)
def test_mft(predict_url, input, label):
    """Minimum Functionality Tests -- Test that model gets correct prediction on sample data."""
    api_body = SimpleModelRequest(review=input).model_dump()
    api_response = requests.post(predict_url, json=api_body)
    response_json = api_response.json()
    prediction = response_json["label"]

    assert label == prediction
