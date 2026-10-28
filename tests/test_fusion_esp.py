import pytest
import requests

from drug_toxicity.fusion_esp import (
    FusionESPResult,
    check_health,
    predict,
    predict_liver_metabolism,
)


class _FakeResponse:
    def __init__(self, text: str, *, status_ok: bool = True) -> None:
        self.text = text
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            msg = "boom"
            raise requests.HTTPError(msg)


def test_predict_parses_interaction(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<p>Prediction results of input sequences interaction; confidence score is [0.87]</p>"
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post", lambda *_a, **_k: _FakeResponse(html)
    )

    result = predict("SEQ", "CCO")

    assert result == FusionESPResult(interacts=True, confidence=0.87)


def test_predict_parses_non_interaction(monkeypatch: pytest.MonkeyPatch) -> None:
    html = (
        "<p>Prediction results of input sequences non-interaction; "
        "confidence score is [1.0117804]</p>"
    )
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post", lambda *_a, **_k: _FakeResponse(html)
    )

    result = predict("SEQ", "CCO")

    assert result == FusionESPResult(interacts=False, confidence=1.0117804)


def test_predict_raises_on_unparseable_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse("<p>nothing useful here</p>"),
    )

    with pytest.raises(ValueError, match="Could not find a prediction result"):
        predict("SEQ", "CCO")


def test_predict_liver_metabolism_resolves_enzyme(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<p>Prediction results of input sequences interaction; confidence score is [0.9]</p>"
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post", lambda *_a, **_k: _FakeResponse(html)
    )

    class _StubResolver:
        def resolve(self, enzyme: str) -> str:
            assert enzyme == "CYP3A4"
            return "RESOLVED_SEQUENCE"

    result = predict_liver_metabolism("CCO", "CYP3A4", resolver=_StubResolver())

    assert result == FusionESPResult(interacts=True, confidence=0.9)


def test_check_health_true_when_response_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<p>Prediction results of input sequences interaction; confidence score is [0.5]</p>"
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post", lambda *_a, **_k: _FakeResponse(html)
    )

    assert check_health() is True


def test_check_health_false_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse("", status_ok=False),
    )

    assert check_health() is False


def test_check_health_false_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_connection_error(*_a: object, **_k: object) -> None:
        msg = "unreachable"
        raise requests.ConnectionError(msg)

    monkeypatch.setattr("drug_toxicity.fusion_esp.requests.post", raise_connection_error)

    assert check_health() is False


def test_check_health_false_on_unparseable_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse("<p>nothing useful here</p>"),
    )

    assert check_health() is False
