from __future__ import annotations

from typing import TYPE_CHECKING

import pubchempy as pcp


def smiles_to_iupac(smiles: str) -> str | None:
    """Return the IUPAC name for a SMILES string via PubChem."""

    try:
        compounds = pcp.get_compounds(
            smiles,
            namespace="smiles",
            as_dataframe=False,
        )

        if not compounds:
            return None

        return compounds[0].iupac_name

    except Exception:
        return None