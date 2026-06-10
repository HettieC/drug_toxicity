from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.error import HTTPError

import pubchempy as pubchem
from molvs import standardize_smiles

logger = logging.getLogger(__name__)


@dataclass
class Validator:
    _cache: set[str] = field(default_factory=set)

    def validate(self, smile_string: str) -> bool:
        try:
            smile_standardized = standardize_smiles(smile_string)

            logger.debug(
                "Standardized SMILES: %s",
                smile_standardized,
            )

            if smile_standardized in self._cache:
                return True

            compounds = pubchem.get_compounds(
                smile_standardized,
                namespace="smiles",
                as_dataframe=False,
            )

            if not isinstance(compounds, list) or not compounds:
                return False

            self._cache.add(smile_standardized)
            return True

        except (ValueError, HTTPError):
            return False
