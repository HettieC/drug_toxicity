from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rdkit import Chem

from . import liver
from .validator import Validator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from rdkit.Chem.rdchem import Mol


@dataclass
class Predictor:
    validator: Validator = field(default_factory=Validator)

    @staticmethod
    def _products_to_smiles(products: list[Mol]) -> list[str]:
        return [Chem.MolToSmiles(x) for x in products]

    def _filter_products(self, products: list[Mol]) -> list[str]:
        return [
            smiles
            for smiles in self._products_to_smiles(products)
            if self.validator.validate(smiles)
        ]

    def predict_phase_i(self, compound: str) -> list[str]:
        products = []
        comp = Chem.MolFromSmiles(compound)

        for reaction in liver.get_phase_i_reactions():
            products.extend(reaction(comp))

        return self._filter_products(products)

    def predict_phase_ii(self, filtered_phase_i_products: Iterable[str]) -> list[str]:
        products = []

        for reactant in filtered_phase_i_products:
            reactant2 = Chem.MolFromSmiles(reactant)
            for reaction in liver.get_phase_ii_reactions():
                products.extend(reaction(reactant2))

        return self._filter_products(products)


def predict_products(compound_smiles: str) -> list[str]:
    predictor = Predictor()
    phase_1 = predictor.predict_phase_i(compound_smiles)
    return phase_1 + predictor.predict_phase_ii(phase_1)
