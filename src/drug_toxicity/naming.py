from __future__ import annotations

from typing import TYPE_CHECKING

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from typing import overload

@overload
def iupac_name(compound: str) -> str | None: ...

@overload
def iupac_name(compound: Mol) -> str | None: ...


def iupac_name(compound: str | Mol) -> str | None:
    if isinstance(compound, Mol):
        compound = Chem.MolToSmiles(compound)

    try:
        compounds = pcp.get_compounds(
            compound,
            namespace="smiles",
            as_dataframe=False,
        )
        return compounds[0].iupac_name if compounds else None
    except Exception:
        return None