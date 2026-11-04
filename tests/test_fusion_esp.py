from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import requests

from drug_toxicity.fusion_esp import (
    FusionESPResult,
    check_health,
    predict,
    predict_file,
    predict_liver_metabolism,
)


class _FakeResponse:
    def __init__(
        self,
        text: str = "",
        *,
        content: bytes | None = None,
        status_ok: bool = True,
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.content = content if content is not None else text.encode()
        self.status_code = status_code
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


def test_predict_retries_on_bad_gateway_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("drug_toxicity.fusion_esp.time.sleep", lambda _seconds: None)
    html = "<p>Prediction results of input sequences interaction; confidence score is [0.87]</p>"
    responses = [
        _FakeResponse(status_code=502),
        _FakeResponse(status_code=502),
        _FakeResponse(html),
    ]
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post", lambda *_a, **_k: responses.pop(0)
    )

    result = predict("SEQ", "CCO")

    assert result == FusionESPResult(interacts=True, confidence=0.87)
    assert responses == []


def test_predict_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("drug_toxicity.fusion_esp.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse(status_code=502, status_ok=False),
    )

    with pytest.raises(requests.HTTPError):
        predict("SEQ", "CCO", max_retries=2)


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


def test_predict_file_returns_results_dataframe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "pairs.xlsx"
    pd.DataFrame({"Protein sequence": ["SEQ"], "SMILES": ["CCO"]}).to_excel(
        input_path, index=False
    )

    report = pd.DataFrame(
        {
            "Protein sequence": ["SEQ"],
            "SMILES": ["CCO"],
            "interaction": ["interaction"],
            "confidence_score": [0.9],
        }
    )
    buffer = BytesIO()
    report.to_excel(buffer, index=False)
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse(content=buffer.getvalue()),
    )

    result = predict_file(input_path)

    pd.testing.assert_frame_equal(result, report)


def test_predict_file_converts_csv_to_xlsx_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "pairs.csv"
    pd.DataFrame({"Protein sequence": ["SEQ"], "SMILES": ["CCO"]}).to_csv(
        input_path, index=False
    )

    report = pd.DataFrame(
        {
            "Protein sequence": ["SEQ"],
            "SMILES": ["CCO"],
            "interaction": ["interaction"],
            "confidence_score": [0.9],
        }
    )
    response_buffer = BytesIO()
    report.to_excel(response_buffer, index=False)
    captured = {}

    def fake_post(*_args: object, files: dict, **_kwargs: object) -> _FakeResponse:
        name, handle = files["Peptide_sequences"]
        captured["name"] = name
        captured["content"] = handle
        return _FakeResponse(content=response_buffer.getvalue())

    monkeypatch.setattr("drug_toxicity.fusion_esp.requests.post", fake_post)

    result = predict_file(input_path)

    pd.testing.assert_frame_equal(result, report)
    assert captured["name"] == "pairs.xlsx"
    uploaded_df = pd.read_excel(BytesIO(captured["content"]))
    pd.testing.assert_frame_equal(
        uploaded_df, pd.DataFrame({"Protein sequence": ["SEQ"], "SMILES": ["CCO"]})
    )


def test_predict_file_strips_servers_stray_index_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "pairs.xlsx"
    pd.DataFrame({"Protein sequence": ["SEQ"], "SMILES": ["CCO"]}).to_excel(
        input_path, index=False
    )

    report = pd.DataFrame(
        {
            "Protein sequence": ["SEQ"],
            "SMILES": ["CCO"],
            "interaction": ["interaction"],
            "confidence_score": [0.9],
        }
    )
    buffer = BytesIO()
    report.to_excel(buffer)  # server writes its own pandas index, uncleaned
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse(content=buffer.getvalue()),
    )

    result = predict_file(input_path)

    assert list(result.columns) == [
        "Protein sequence",
        "SMILES",
        "interaction",
        "confidence_score",
    ]


def test_predict_file_raises_on_unparseable_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "pairs.xlsx"
    pd.DataFrame({"Protein sequence": ["SEQ"], "SMILES": ["CCO"]}).to_excel(
        input_path, index=False
    )
    monkeypatch.setattr(
        "drug_toxicity.fusion_esp.requests.post",
        lambda *_a, **_k: _FakeResponse("<p>no results</p>"),
    )

    with pytest.raises(ValueError, match="did not return a results file"):
        predict_file(input_path)


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
