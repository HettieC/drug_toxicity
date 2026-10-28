import pytest

from drug_toxicity.liver_enzymes import LiverEnzymeResolver

SEQUENCE = "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYG"


def test_resolve_raw_sequence_passes_through() -> None:
    resolver = LiverEnzymeResolver()
    assert resolver.resolve(SEQUENCE) == SEQUENCE


def test_resolve_uniprot_accession(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = LiverEnzymeResolver()
    monkeypatch.setattr(
        resolver, "_fetch_sequence_by_accession", lambda _accession: SEQUENCE
    )

    assert resolver.resolve("P08684") == SEQUENCE


def test_resolve_curated_gene_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = LiverEnzymeResolver()
    monkeypatch.setattr(
        resolver, "_fetch_accession_by_gene_symbol", lambda _gene_symbol: "P08684"
    )
    monkeypatch.setattr(
        resolver, "_fetch_sequence_by_accession", lambda _accession: SEQUENCE
    )

    assert resolver.resolve("cyp3a4") == SEQUENCE


def test_resolve_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = LiverEnzymeResolver()
    calls = []
    monkeypatch.setattr(
        resolver,
        "_fetch_sequence_by_accession",
        lambda accession: calls.append(accession) or SEQUENCE,
    )

    resolver.resolve("P08684")
    resolver.resolve("P08684")

    assert calls == ["P08684"]


def test_resolve_unknown_identifier_raises() -> None:
    resolver = LiverEnzymeResolver()

    with pytest.raises(ValueError, match="not a known liver enzyme"):
        resolver.resolve("NOT_A_GENE")
