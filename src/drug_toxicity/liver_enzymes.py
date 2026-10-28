"""Resolve human liver drug-metabolising enzymes to amino acid sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

#: Human Phase I/II drug-metabolising enzymes, mapped to the reaction
#: categories they correspond to in :mod:`drug_toxicity.liver`. Gene symbols
#: only - accessions are resolved live against UniProt so a stale hardcoded
#: ID can never silently point at the wrong protein.
LIVER_ENZYMES: dict[str, str] = {
    # Phase I - cytochrome P450s
    "CYP1A2": "aromatic_hydroxylation",
    "CYP2C9": "aliphatic_hydroxylation",
    "CYP2C19": "aliphatic_hydroxylation",
    "CYP2D6": "aliphatic_hydroxylation",
    "CYP2E1": "aliphatic_hydroxylation",
    "CYP3A4": "aliphatic_hydroxylation",
    "CYP3A5": "aliphatic_hydroxylation",
    "FMO3": "n_oxidation",
    "ADH1B": "alcohol_oxidation",
    "ALDH2": "alcohol_dehydrogenation",
    "EPHX1": "hydrolysis_epoxide",
    "CBR1": "carbonyl_reduction",
    "NQO1": "nitro_reduction1",
    # Phase II
    "UGT1A1": "o_glucuronidation_phenols_alcohols",
    "UGT1A3": "n_glucuronidation_amines",
    "UGT1A4": "n_glucuronidation_amines",
    "UGT1A6": "o_glucuronidation_phenols_alcohols",
    "UGT1A9": "o_glucuronidation_carboxylic_acids",
    "UGT2B7": "o_glucuronidation_phenols_alcohols",
    "UGT2B15": "o_glucuronidation_phenols_alcohols",
    "SULT1A1": "sulfonation_alcohols",
    "SULT1A3": "sulfonation_alcohols",
    "SULT2A1": "sulfonation_alcohols",
    "GSTA1": "gluthationylation_expoxides",
    "GSTM1": "gluthationylation_expoxides",
    "GSTP1": "gluthationylation_alpha_beta_unsaturated_carbonlys",
    "GSTT1": "gluthationylation_alkyl_halides",
    "COMT": "methylation_catechols",
    "TPMT": "methylation_thiopurines",
    "NNMT": "methylation_amines",
    "NAT1": "acetylation_first_amines",
    "NAT2": "acetylation_first_amines",
    "GLYAT": "conjugation_glycine",
}

_UNIPROT_ACCESSION = re.compile(
    r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
)
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")
_UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb"
_MIN_SEQUENCE_LENGTH = 20


def _looks_like_sequence(value: str) -> bool:
    cleaned = value.upper()
    return len(cleaned) >= _MIN_SEQUENCE_LENGTH and set(cleaned) <= _AMINO_ACIDS


@dataclass
class LiverEnzymeResolver:
    """Resolve a liver enzyme name, UniProt accession, or raw sequence to a sequence."""

    _cache: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0

    def resolve(self, enzyme: str) -> str:
        """Resolve a gene symbol, UniProt accession, or raw sequence to a sequence."""
        key = enzyme.strip()

        if key in self._cache:
            return self._cache[key]

        if _looks_like_sequence(key):
            sequence = key.upper()
        elif _UNIPROT_ACCESSION.fullmatch(key.upper()):
            sequence = self._fetch_sequence_by_accession(key.upper())
        else:
            gene_symbol = key.upper()
            if gene_symbol not in LIVER_ENZYMES:
                msg = (
                    f"{enzyme!r} is not a known liver enzyme gene symbol, a "
                    "UniProt accession, or a raw amino acid sequence."
                )
                raise ValueError(msg)
            accession = self._fetch_accession_by_gene_symbol(gene_symbol)
            sequence = self._fetch_sequence_by_accession(accession)

        self._cache[key] = sequence
        return sequence

    def _fetch_accession_by_gene_symbol(self, gene_symbol: str) -> str:
        response = requests.get(
            f"{_UNIPROT_BASE_URL}/search",
            params={
                "query": f"gene:{gene_symbol} AND organism_id:9606 AND reviewed:true",
                "fields": "accession",
                "format": "json",
                "size": 1,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            msg = f"No reviewed human UniProt entry found for gene {gene_symbol!r}"
            raise ValueError(msg)

        return results[0]["primaryAccession"]

    def _fetch_sequence_by_accession(self, accession: str) -> str:
        response = requests.get(
            f"{_UNIPROT_BASE_URL}/{accession}.fasta", timeout=self.timeout
        )
        response.raise_for_status()
        return "".join(response.text.splitlines()[1:])
