import pytest

from src.events.schema import decode_event, encode_event


def test_event_round_trip_is_validated() -> None:
    event = {"user_idx": 4, "item_idx": 9, "event_type": "click", "event_time": 123}
    assert decode_event(encode_event(event)) == event


@pytest.mark.parametrize(
    "event",
    [
        {"user_idx": -1, "item_idx": 2, "event_type": "click", "event_time": 1},
        {"user_idx": 1, "item_idx": 2, "event_type": "purchase", "event_time": 1},
        {"user_idx": True, "item_idx": 2, "event_type": "view", "event_time": 1},
    ],
)
def test_event_rejects_invalid_values(event) -> None:
    with pytest.raises(ValueError):
        encode_event(event)


def test_event_decoder_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        decode_event(b"not-json")
