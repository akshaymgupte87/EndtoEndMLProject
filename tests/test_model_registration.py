import pytest

from src.models.register_model import check_validation_gate


def _report(split="validation", model_recall=0.025, model_ndcg=0.012):
    return {
        "split": split,
        "metrics_at_k": {
            "two_tower": {"10": {"recall": model_recall, "ndcg": model_ndcg}},
            "popularity_baseline_on_same_user_sample": {
                "10": {"recall": 0.02, "ndcg": 0.01}
            },
        },
    }


def test_model_registration_gate_accepts_validation_gain() -> None:
    result = check_validation_gate(_report())
    assert result["ndcg_lift"] == pytest.approx(0.2)


def test_model_registration_gate_rejects_test_report_and_regression() -> None:
    with pytest.raises(ValueError, match="only a validation report"):
        check_validation_gate(_report(split="test"))
    with pytest.raises(ValueError, match="Recall@10"):
        check_validation_gate(_report(model_recall=0.01))


def test_model_registration_gate_rejects_non_finite_metrics() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        check_validation_gate(_report(model_ndcg=float("nan")))
