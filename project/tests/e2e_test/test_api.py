import pytest
import requests

from src.constants import SentimentLabel
from src.data_models import SimpleModelRequest


@pytest.mark.parametrize(
    "review",
    [
        "This product is amazing!",
        "I'm not sure about this one.",
        "Terrible experience, would not recommend.",
    ],
)
def test_predict_endpoint_with_different_reviews(predict_url, review):
    """
    Note that this method currently only tests that the prediction works
    with an API. It DOES NOT test the prediction results are correct!
    When we improve our models enough so that it predicts correct results reliably
    then we can change this test to test prediction results correctness as well.
    """
    api_body = SimpleModelRequest(review=review).model_dump()
    api_response = requests.post(predict_url, json=api_body)
    response_json = api_response.json()
    print(response_json)

    assert api_response.status_code == 200
    assert response_json["label"] in [label.value for label in SentimentLabel]
    assert 0 <= response_json["score"] <= 1
