"""Client for the FusionESP enzyme-substrate prediction webserver.

FusionESP (https://github.com/dzjxzyd/FusionESP) is hosted as a small Flask
app with no documented API, only an HTML form. This parses its POST response
rather than calling a stable, documented contract, so the hosted URL or
response format may change without notice.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from drug_toxicity.liver_enzymes import LiverEnzymeResolver

DEFAULT_URL = "https://rqkjkgpsyu.us-east-1.awsapprunner.com"

#: Columns required in a batch-prediction input file, per the webserver's
#: "Large-scale output version" (see /pred_with_file in the FusionESP app).
BATCH_PROTEIN_COLUMN = "Protein sequence"
BATCH_SMILES_COLUMN = "SMILES"

_RESULT_PATTERN = re.compile(
    r"Prediction results of input sequences (?P<label>interaction|non-interaction);"
    r"\s*confidence score is \[(?P<score>[^\]]+)\]"
)

#: The hosted instance intermittently returns these under load - e.g. a
#: large batch file keeping the backend busy long enough for AWS App
#: Runner's proxy to report it as down - unrelated to the request itself.
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


def _post_with_retries(
    url: str, *, max_retries: int, backoff_seconds: float = 5.0, **kwargs: Any
) -> requests.Response:
    """POST, retrying with backoff on the webserver's transient gateway errors."""
    for attempt in range(max_retries + 1):
        response = requests.post(url, **kwargs)  # noqa: S113 - callers always pass timeout
        if response.status_code not in _RETRYABLE_STATUS_CODES or attempt == max_retries:
            return response
        time.sleep(backoff_seconds * (2**attempt))
    return response


@dataclass
class FusionESPResult:
    """A FusionESP prediction for one enzyme/substrate pair."""

    interacts: bool
    confidence: float


def predict(
    enzyme_sequence: str,
    smiles: str,
    base_url: str = DEFAULT_URL,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> FusionESPResult:
    """Score one enzyme/substrate pair against the FusionESP webserver."""
    response = _post_with_retries(
        f"{base_url}/predict",
        max_retries=max_retries,
        data={"Enzyme_sequence": enzyme_sequence, "molecule_SMILES": smiles},
        timeout=timeout,
    )
    response.raise_for_status()

    match = _RESULT_PATTERN.search(response.text)
    if match is None:
        msg = "Could not find a prediction result in the FusionESP response"
        raise ValueError(msg)

    return FusionESPResult(
        interacts=match["label"] == "interaction",
        confidence=float(match["score"]),
    )


_CSV_SUFFIXES = {".csv"}


def predict_file(
    file_path: str | Path,
    base_url: str = DEFAULT_URL,
    timeout: float = 300.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Score many enzyme/substrate pairs from a csv/xls/xlsx file.

    `file_path` must have `"Protein sequence"` and `"SMILES"` columns, one
    pair per row (see `BATCH_PROTEIN_COLUMN`/`BATCH_SMILES_COLUMN`). The
    webserver itself only accepts xls/xlsx, so a `.csv` file is converted to
    xlsx in memory before upload. Returns those columns alongside the
    webserver's `"interaction"` and `"confidence_score"` columns, one row
    per input pair.
    """
    path = Path(file_path)

    if path.suffix.lower() in _CSV_SUFFIXES:
        upload_name = f"{path.stem}.xlsx"
        buffer = BytesIO()
        pd.read_csv(path).to_excel(buffer, index=False)
        upload_content = buffer.getvalue()
    else:
        upload_name = path.name
        upload_content = path.read_bytes()

    response = _post_with_retries(
        f"{base_url}/pred_with_file",
        max_retries=max_retries,
        files={"Peptide_sequences": (upload_name, upload_content)},
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        results = pd.read_excel(BytesIO(response.content))
    except ValueError as error:
        msg = (
            "FusionESP did not return a results file - check that "
            f"{file_path!r} has '{BATCH_PROTEIN_COLUMN}' and "
            f"'{BATCH_SMILES_COLUMN}' columns with at least one row."
        )
        raise ValueError(msg) from error

    return results.loc[:, ~results.columns.str.match(r"^Unnamed: \d+$")]


#: Arbitrary but valid inputs used only to confirm the service responds in the
#: expected format - not meant to be biologically meaningful.
_HEALTH_CHECK_SEQUENCE = "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGN"
_HEALTH_CHECK_SMILES = "CCO"


def check_health(base_url: str = DEFAULT_URL, timeout: float = 30.0) -> bool:
    """Check that the FusionESP webserver is up and still returns parseable results.

    The service exposes no status/health endpoint, so this runs a throwaway
    prediction and reports whether it both succeeds and still matches the
    expected response format.
    """
    try:
        predict(
            _HEALTH_CHECK_SEQUENCE,
            _HEALTH_CHECK_SMILES,
            base_url=base_url,
            timeout=timeout,
        )
    except (requests.RequestException, ValueError):
        return False

    return True


def predict_liver_metabolism(
    product_smiles: str,
    enzyme: str,
    resolver: LiverEnzymeResolver | None = None,
    base_url: str = DEFAULT_URL,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> FusionESPResult:
    """Score whether `product_smiles` is a substrate of a human liver enzyme.

    `enzyme` may be a curated gene symbol (see `liver_enzymes.LIVER_ENZYMES`),
    a UniProt accession, or a raw amino acid sequence.
    """
    resolver = resolver or LiverEnzymeResolver()
    sequence = resolver.resolve(enzyme)
    return predict(
        sequence, product_smiles, base_url=base_url, timeout=timeout, max_retries=max_retries
    )
