"""The HTTP body and the batch rows are validated by the same pydantic base schema."""

import pytest
from pydantic import ValidationError

from serving.schemas import BatchOptions, BatchRecord, PredictRequest, RecommendationQuery


def test_both_inputs_inherit_the_shared_schema():
    assert issubclass(PredictRequest, RecommendationQuery)
    assert issubclass(BatchRecord, RecommendationQuery)
    assert {"user_id", "genre"} <= set(PredictRequest.model_fields) & set(BatchRecord.model_fields)


@pytest.mark.parametrize(
    "user_id, genre",
    [(0, None), (-4, None), (11, "x" * 51)],
)
def test_the_same_bad_values_are_rejected_over_http_and_in_batch(user_id, genre):
    with pytest.raises(ValidationError) as http:
        PredictRequest(user_id=user_id, genre=genre)
    with pytest.raises(ValidationError) as batch:
        BatchRecord(record_id="r", user_id=str(user_id), genre=genre)
    assert {e["loc"] for e in http.value.errors()} == {e["loc"] for e in batch.value.errors()}


def test_http_is_strict_but_batch_parses_digit_text():
    with pytest.raises(ValidationError):
        PredictRequest(user_id="11")
    assert BatchRecord(record_id="r", user_id=" 11 ").user_id == 11
    for bad in ("11.0", "-5", "abc", "", "\uff11\uff11", "1e3", "0"):
        with pytest.raises(ValidationError):
            BatchRecord(record_id="r", user_id=bad)


def test_unknown_fields_are_rejected_by_both():
    with pytest.raises(ValidationError):
        PredictRequest(user_id=1, extra="x")
    with pytest.raises(ValidationError):
        BatchRecord(record_id="r", user_id="1", extra="x")


def test_blank_genre_in_a_batch_row_means_no_filter():
    assert BatchRecord(record_id="r", user_id="1", genre="   ").genre is None
    assert BatchRecord(record_id="r", user_id="1", genre=" Drama ").genre == "Drama"


def test_batch_options_limit_k():
    assert BatchOptions(k=100).k == 100
    for bad in (0, 101, -1):
        with pytest.raises(ValidationError):
            BatchOptions(k=bad)
