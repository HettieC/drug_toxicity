from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rdkit import Chem

from . import liver
from .liver_enzymes import REACTION_ENZYMES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rdkit.Chem.rdchem import Mol


@dataclass(frozen=True)
class MetabolicProduct:
    """A predicted metabolite, with the reaction and candidate genes behind it.

    Frozen and hashable so callers can deduplicate predictions with `set()`,
    as the existing docs notebook does.
    """

    smiles: str
    reaction: str
    genes: tuple[str, ...] = ()


@dataclass
class Predictor:
    def predict_phase_i(self, compound: str) -> list[MetabolicProduct]:
        products = []
        comp = Chem.MolFromSmiles(compound)

        for reaction in liver.get_phase_i_reactions():
            products.extend(self._run_reaction(reaction, comp))

        return products

    def predict_phase_ii(
        self, filtered_phase_i_products: Iterable[MetabolicProduct]
    ) -> list[MetabolicProduct]:
        products = []

        for reactant in filtered_phase_i_products:
            reactant2 = Chem.MolFromSmiles(reactant.smiles)
            for reaction in liver.get_phase_ii_reactions():
                products.extend(self._run_reaction(reaction, reactant2))

        return products

    @staticmethod
    def _run_reaction(reaction: Callable, comp: Mol) -> list[MetabolicProduct]:
        name = reaction.__name__
        genes = tuple(REACTION_ENZYMES.get(name, []))
        return [
            MetabolicProduct(smiles=smiles, reaction=name, genes=genes)
            for smiles in reaction(comp)
        ]


def predict_products(compound_smiles: str) -> list[MetabolicProduct]:
    predictor = Predictor()
    phase_1 = predictor.predict_phase_i(compound_smiles)
    return phase_1 + predictor.predict_phase_ii(phase_1)
