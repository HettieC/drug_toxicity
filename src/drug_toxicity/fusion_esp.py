"""Client for the FusionESP enzyme-substrate prediction webserver.

FusionESP (https://github.com/dzjxzyd/FusionESP) is hosted as a small Flask
app with no documented API, only an HTML form. This parses its POST response
rather than calling a stable, documented contract, so the hosted URL or
response format may change without notice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from drug_toxicity.liver_enzymes import LiverEnzymeResolver

DEFAULT_URL = "https://rqkjkgpsyu.us-east-1.awsapprunner.com"

_RESULT_PATTERN = re.compile(
    r"Prediction results of input sequences (?P<label>interaction|non-interaction);"
    r"\s*confidence score is \[(?P<score>[^\]]+)\]"
)


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
) -> FusionESPResult:
    """Score one enzyme/substrate pair against the FusionESP webserver."""
    response = requests.post(
        f"{base_url}/predict",
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
) -> FusionESPResult:
    """Score whether `product_smiles` is a substrate of a human liver enzyme.

    `enzyme` may be a curated gene symbol (see `liver_enzymes.LIVER_ENZYMES`),
    a UniProt accession, or a raw amino acid sequence.
    """
    resolver = resolver or LiverEnzymeResolver()
    sequence = resolver.resolve(enzyme)
    return predict(sequence, product_smiles, base_url=base_url, timeout=timeout)
