from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pubchempy as pubchem
from pubchempy import BadRequestError
from molvs import standardize_smiles

logger = logging.getLogger(__name__)


@dataclass
class Validator:
    _cache: set[str] = field(default_factory=set)

    def validate(self, smile_string: str) -> bool:
        try:
            smile_standardized = standardize_smiles(smile_string)

            if smile_standardized in self._cache:
                return True

            compounds = pubchem.get_compounds(
                smile_standardized,
                namespace="smiles",
                as_dataframe=False,
            )

            if not compounds:
                return False

            self._cache.add(smile_standardized)
            return True

        except BadRequestError:
            logger.debug(
                "PubChem rejected structure: %s",
                smile_string,
            )
            return False

        except ValueError:
            return False