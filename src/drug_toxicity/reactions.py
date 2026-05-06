from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rdkit import Chem, DataStructs
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from rdkit.Chem.rdmolfiles import MolFromSmarts

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rdkit.Chem.rdchem import Mol


class Reactions:
    water = "[#8]"
    glucuronic_acid = "[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
    sulfon = "[#16](=[#8])(=[#8])-[#8-]"
    glutathione = "[#16]-[#6](-[#7]-[#6](=[#8])-[#6]-[#6]-[#6](-[#7+])-[#6](=[#8])-[#8-])-[#6](=[#8])-[#7]-[#6]-[#6](=[#8])-[#8]"
    methyl_group = "[#6]"
    acetyl_group = "[#6](=[#8])-[#6]"

    water_mol = Chem.MolFromSmarts(water)
    glucuronic_acid_mol = MolFromSmarts(glucuronic_acid)

    @staticmethod
    def _extract_touples(touples: Iterable[Iterable[Mol]]) -> list[Mol]:
        return [y for x in touples for y in x]

    @classmethod
    def _perform_reaction(
        cls, comp1: Mol, comp2: Mol, react1: str, react2: str, product: str
    ) -> list[Mol]:
        reaction = react1 + "." + react2 + ">>" + product
        test: ChemicalReaction = ReactionFromSmarts(reaction)
        substrates = test.RunReactants((comp1, comp2))
        extracted_substrates = cls._extract_touples(substrates)
        is_duplicate = False
        final_substrates = [extracted_substrates[0]]

        for i in range(1, len(extracted_substrates)):
            for j in range(len(final_substrates)):
                comparison_score = DataStructs.FingerprintSimilarity(
                    Chem.RDKFingerprint(extracted_substrates[i]),
                    Chem.RDKFingerprint(final_substrates[j]),
                    metric=DataStructs.TanimotoSimilarity,
                )
                if math.isclose(comparison_score, 1):
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_substrates.append(extracted_substrates[i])

            is_duplicate = False

        return final_substrates

    @classmethod
    def aliphatic_hydroxylation(cls, comp: Mol) -> list[Mol]:
        """Perform aliphatic hydroxylation"""
        ethane = "[#6:1]-[#6:2]"
        ethanol = "[#6:1]-[#6:2]-[#8]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(ethane))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, ethane, cls.water, ethanol
            )

        return final_substrates

    @classmethod
    def aromatic_hydroxylation(cls, comp: Mol) -> list[Mol]:
        """Perform aromatic hydroxylation"""
        benzene = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[#6:5]:[#6:6]:1"
        alkyl_phenol = "[#6:6]([#8#1]):1[#6:5]:[#6:4]:[#6:3]:[#6:2]:[#6:1]:1"

        canonicalized_smiles = Chem.MolToSmiles(comp, isomericSmiles=True)
        canonicalized_comp = Chem.MolFromSmiles(canonicalized_smiles)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(benzene))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                canonicalized_comp, cls.water_mol, benzene, cls.water, alkyl_phenol
            )

        return final_substrates

    @classmethod
    def n_oxidation(cls, comp: Mol) -> list[Mol]:
        """Perform n oxidation"""
        primary_amine = "[#7:2]-[#6:1]"
        n_oxide = "[#6:1]-[#7:2]([#8][#1])"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(primary_amine))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, primary_amine, cls.water, n_oxide
            )

        return final_substrates

    @classmethod
    def s_oxidation(cls, comp: Mol) -> list[Mol]:
        """Perform s oxidation"""
        thioeter = "[#6:1]-[#16:2]-[#6:3]"
        sulfoxide = "[#6:1]-[#16:2](=[#8])-[#6:3]"
        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(thioeter))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, thioeter, cls.water, sulfoxide
            )

        return final_substrates

    @classmethod
    def n_dealkylation(cls, comp: Mol) -> list[Mol]:
        """Perform n dealkylation"""
        secondary_amine_no_ring = "[#6:1]!@[#7:2]!@[#6:3]"

        # primary Amine + Adelyhde
        product = "[#7:2]-[#6:1]" + "." + "[#6:3](=[#8])"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(secondary_amine_no_ring))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, secondary_amine_no_ring, cls.water, product
            )

        return final_substrates

    @classmethod
    def o_dealkylation(cls, comp: Mol) -> list[Mol]:
        """Perform o dealkylation"""
        ether = "[#6:1]-[#8:2]-[#6:3]-[#6:4]"
        product = "[#6:1]-[#8:2]" + "." + "[#6:4]-[#6:3](=[#8])"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(ether))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, ether, cls.water, product
            )

        return final_substrates

    @classmethod
    def epoxidation(cls, comp: Mol) -> list[Mol]:
        """Perform epoxidation"""
        alkene = "[#6:1]-[#6:2]=[#6:3]-[#6:4]"
        epoxyde = "[#6:1]-[#6:2]-1[#8]-[#6:3]:1-[#6:4]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(alkene))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, alkene, cls.water, epoxyde
            )

        return final_substrates

    @classmethod
    def alcohol_oxidation(cls, comp: Mol) -> list[Mol]:
        """Perform alcohol oxidation"""
        ethanol = "[#6:1]-[#6:2]-[#8]"
        aldehyde = "[#6:1]-[#6:2](=[#8])"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(ethanol))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, ethanol, cls.water, aldehyde
            )

        return final_substrates

    @classmethod
    def oxidative_deamination(cls, comp: Mol) -> list[Mol]:
        """Perform oxidative deamination"""
        primary_amine = "[#7:2]-[#6:1]"
        # Aldehyde + Ammonia
        product = "[#6:1](=[#8])" + "." + "[#7:2]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(primary_amine))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, primary_amine, cls.water, product
            )

        return final_substrates

    @classmethod
    def decarboxylation(cls, comp: Mol) -> list[Mol]:
        """Perform decarboxylation"""
        propionic_acid = "[#6:1](=[#8])(-[#8])-[#6:2]-[#6:3]"
        product = "[#6:2]-[#6:3]" + "." + "[#6:1](=[#8])(=[#8])"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(propionic_acid))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, propionic_acid, cls.water, product
            )

        return final_substrates

    @classmethod
    def phenolamine_oxidation(cls, comp: Mol) -> list[Mol]:
        """Perform phenolamine oxidation"""
        amino_phenol = "[#6:1](-[#7:8]):1:[#6:2]:[#6:3]:[#6:4](-[#8:7]):[#6:5]:[#6:6]:1"
        quinone_imine = (
            "[#6:1](=[#7:8])-1-[#6:2]=[#6:3]-[#6:4](=[#8:7])-[#6:5]=[#6:6]-1"
        )

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(amino_phenol))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, amino_phenol, cls.water, quinone_imine
            )

        return final_substrates

    @classmethod
    def hydrolysis_ester(cls, comp: Mol) -> list[Mol]:
        """Perform hydrolysis ester"""
        ester = "[#6:1]-[#6:2](=[#8:3])(-[#8:5]-[#6:6])"
        product = "[#6:6]-[#8:5]" + "." + "[#6:1]-[#6:2](=[#8:3])-[#8]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(ester))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, ester, cls.water, product
            )

        return final_substrates

    @classmethod
    def deacetylation(cls, comp: Mol) -> list[Mol]:
        """Perform deacetylation"""
        acetamide = "[#6:1]-[#6:2](=[#8:3])(-[#7:4](-[#6:5]))"
        # acetic_acid + primary amin
        product = "[#6:1]-[#6:2](=[#8:3])-[#8]" + "." + "[#7:4]-[#6:5]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(acetamide))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, acetamide, cls.water, product
            )

        return final_substrates

    @classmethod
    def hydrolysis_aliphatic(cls, comp: Mol) -> list[Mol]:
        """Perform aliphatic hydrolysis"""
        aliphatic = "[#6:1]-&!@[#6,#7:2]"
        product = "[#6:1]-[#8]" + "." + "[#6,#7:2]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(aliphatic))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, aliphatic, cls.water, product
            )

        return final_substrates

    @classmethod
    def alcohol_dehydrogenation(cls, comp: Mol) -> list[Mol]:
        """Perform alcohol dehydrogenation"""
        alcohol = "[#6:1]-[#8:2]"
        dehydrogenated_alcohol = "[#6:1]=[#8:2]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(alcohol))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, alcohol, cls.water, dehydrogenated_alcohol
            )

        return final_substrates

    @classmethod
    def nitro_reduction1(cls, comp: Mol) -> list[Mol]:
        """Perform nitrophenol reduction"""
        nitrophenol = (
            "[#6:1]:1(-[#7+:7](-[#8-])(=[#8])):[#6:2]:[#6:3]:[#6:4]:[#6:5]:[#6:6]:1"
        )
        aniline = "[#6:1]:1(-[#7:7]):[#6:2]:[#6:3]:[#6:4]:[#6:5]:[#6:6]:1"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(nitrophenol))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, nitrophenol, cls.water, aniline
            )

        return final_substrates

    @classmethod
    def nitro_reduction2(cls, comp: Mol) -> list[Mol]:
        """Perform nitrophenol reduction"""
        nitrophenol = (
            "[#6:1]-1(=[#7+:7](-[#8-])(-[#8-]))-[#6:2]=[#6:3]-[#6:4]-[#6:5]=[#6:6]-1"
        )
        aniline = "[#6:1]:1(-[#7:7]):[#6:2]:[#6:3]:[#6:4]:[#6:5]:[#6:6]:1"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(nitrophenol))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, nitrophenol, cls.water, aniline
            )

        return final_substrates

    @classmethod
    def hydrolysis_amide(cls, comp: Mol) -> list[Mol]:
        """Perform amide hydrolysis"""
        carbon_acid_amide = "[#6:1]-[#6:2](=[#8:3])-[#7:4](-[#6:5])"
        product = "[#6:1]-[#6:2](=[#8:3])-[#8]" + "." + "[#7:4](-[#6:5])"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(carbon_acid_amide))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, carbon_acid_amide, cls.water, product
            )

        return final_substrates

    @classmethod
    def hydrolysis_epoxide(cls, comp: Mol) -> list[Mol]:
        """Perform epoxide hydrolysis"""
        epoxide = "[#6:2](-[#6:1])-1[#8:3]-[#6:4]-1(-[#6:5])"
        hydrolysed_epoxide = "[#6:2]-[#6:1](-[#8:3])-[#6:4](-[#8])-[#6:5]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(epoxide))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, epoxide, cls.water, hydrolysed_epoxide
            )

        return final_substrates

    @classmethod
    def dehalogenation(cls, comp: Mol) -> list[Mol]:
        """Perform dehalogenation"""
        halogen = "[#6:1]-[F,Cl,Br,I:2]"
        product = "[#6:1]" + "." + "[F,Cl,Br,I:2]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(halogen))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, halogen, cls.water, product
            )

        return final_substrates

    @classmethod
    def carbonyl_reduction(cls, comp: Mol) -> list[Mol]:
        """Perform carbonyl reduction"""
        carbonyl = "[#6:1]-[#6:2](=[#8:4])-[#6:3]"
        alcohol = "[#6:1]-[#6:2](-[#8:4])-[#6:3]"

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(carbonyl))

        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, cls.water_mol, carbonyl, cls.water, alcohol
            )

        return final_substrates

    # PhaseII
    # Glucuronidation
    @classmethod
    def o_glucuronidation_phenols_alcohols(cls, comp: Mol) -> list[Mol]:
        """Perform O-glucuronidation of phenols and alcohols"""
        methanol = "[#6:1]-[#8:2]"
        glucuronic_acid = "[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
        product = "[#6](-[#8:2](-[#6:1]))-1[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
        react2 = Chem.MolFromSmarts(glucuronic_acid)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(methanol))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, methanol, glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def o_glucuronidation_carboxylic_acids(cls, comp: Mol) -> list[Mol]:
        """Perform O-glucuronidation of carboxylic acids"""
        acid = "[#6:1](=[#8])-[#8]"
        glucuronic_acid = "[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
        product = "[#6:1](=[#8])-[#8]-[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
        react2 = Chem.MolFromSmarts(glucuronic_acid)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(acid))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, acid, glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def n_glucuronidation_amines(cls, comp: Mol) -> list[Mol]:
        """Perform N-glucuronidation of amines"""
        matches_1 = comp.GetSubstructMatches(Chem.MolFromSmarts("[#7]-[#6]"))
        matches_2 = comp.GetSubstructMatches(Chem.MolFromSmarts("[#7](-[#6])(-[#6])"))
        matches_3 = comp.GetSubstructMatches(
            Chem.MolFromSmarts("[#6]-[#7](-[#6])(-[#6])")
        )

        final_substrates = []

        if len(matches_3) > 0:
            amine = "[#6:3]-[#7:4](-[#6:5])(-[#6:6])"
            product = "[#6](-[+#7:4](-[#6:3])(-[#6:5])(-[#6:6]))-1[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
            final_substrates = cls._perform_reaction(
                comp, cls.glucuronic_acid_mol, amine, cls.glucuronic_acid, product
            )

        elif len(matches_2) > 0:
            amine = "[#6:3]-[#7:4](-[#6:5])"
            product = "[#6](-[#7:4](-[#6:3])(-[#6:5]))-1[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
            final_substrates = cls._perform_reaction(
                comp, cls.glucuronic_acid_mol, amine, cls.glucuronic_acid, product
            )

        elif len(matches_1) > 0:
            amine = "[#6:3]-[#7:4]"
            product = "[#6](-[#7:4](-[#6:3]))-1[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
            final_substrates = cls._perform_reaction(
                comp, cls.glucuronic_acid_mol, amine, cls.glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def n_glucuronidation_pyridine(cls, comp: Mol) -> list[Mol]:
        """Perform N-glucuronidation of pyridines"""
        pyridine = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[#7:5]:[#6:6]:1"
        product = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[+#7:5](-[#6]-2[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-2):[#6:6]:1"
        react2 = Chem.MolFromSmarts(cls.glucuronic_acid)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(pyridine))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, pyridine, cls.glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def s_glucuronidation_thiols(cls, comp: Mol) -> list[Mol]:
        """Perform S-glucuronidation of thiols"""
        thiols = "[#16:1]"
        product = "[#16:1]-[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
        react2 = Chem.MolFromSmarts(cls.glucuronic_acid)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(thiols))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, thiols, cls.glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def glucuronidation_alpha_diketones(cls, comp: Mol) -> list[Mol]:
        """Perform C-glucuronidation at the alpha position of diketones"""
        diketones = "[#6:1]-[#6](=[#8])-[#6:2](-[#6:3])-[#6](=[#8])-[#6:4]"
        product = "[#6:1]-[#6](=[#8])-[#6:2](-[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1)(-[#6:3])-[#6](=[#8])-[#6:4]"
        react2 = Chem.MolFromSmarts(cls.glucuronic_acid)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(diketones))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, diketones, cls.glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def glucuronidation_alkynes(cls, comp: Mol) -> list[Mol]:
        """Perform C-glucuronidation of alkynes"""
        alkene = "[#6:1]#[#6:2]"
        product = "[#6:1]#[#6:2]-[#6]-1[#6](-[#8]-[#1])-[#6](-[#8][#1])-[#6](-[#8][#1])-[#6](-[#6](=[#8])(-[#8][#1]))-[#8]-1"
        react2 = Chem.MolFromSmarts(cls.glucuronic_acid)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(alkene))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, alkene, cls.glucuronic_acid, product
            )

        return final_substrates

    # Suflonation

    @classmethod
    def sulfonation_alcohols(cls, comp: Mol) -> list[Mol]:
        """Perform sulfonation of alcohols"""
        alcohol = "[#6:1]-[#8:2]"
        product = "[#6:1]-[#8:2]-[#16](=[#8])([#8])-[#8-]"
        react2 = Chem.MolFromSmarts(cls.sulfon)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(alcohol))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, alcohol, cls.sulfon, product
            )

        return final_substrates

    @classmethod
    def sulfonation_aromatic_n_oxides(cls, comp: Mol) -> list[Mol]:
        """Perform sulfonation of aromatic N-oxides"""
        n_oxide = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[#7+:5](-[#8-:7]):[#6:6]:1"
        product = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[#7+](-[#8]-[#16](=[#8])([#8])-[#8-]):[#6:6]:1"
        react2 = Chem.MolFromSmarts(cls.sulfon)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(n_oxide))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, n_oxide, cls.sulfon, product
            )

        return final_substrates

    # Glutathione
    @classmethod
    def gluthationylation_expoxides(cls, comp: Mol) -> list[Mol]:
        """Perform gluthationylation of expoxides"""
        expoxides = "[#8:2]:1:[#6:1]:[#6:3]:1"
        product = "[#6:1](-[#8:2])-[#6:3]-[#16]-[#6](-[#7]-[#6](=[#8])-[#6]-[#6]-[#6](-[#7+])-[#6](=[#8])-[#8-])-[#6](=[#8])-[#7]-[#6]-[#6](=[#8])-[#8]"
        react2 = Chem.MolFromSmarts(cls.glutathione)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(expoxides))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, expoxides, cls.glutathione, product
            )

        return final_substrates

    @classmethod
    def gluthationylation_alpha_beta_unsaturated_carbonlys(cls, comp: Mol) -> list[Mol]:
        """Perform gluthationylation of alpha,beta-unsaturated carbonlys"""
        carbonyls = "[#6:1]=[#6:2]-[#6:3](=[#8])"
        product = "[#6:1](-[#16]-[#6](-[#7]-[#6](=[#8])-[#6]-[#6]-[#6](-[#7+])-[#6](=[#8])-[#8-])-[#6](=[#8])-[#7]-[#6]-[#6](=[#8])-[#8])=[#6:2]-[#6:3](=[#8])"
        react2 = Chem.MolFromSmarts(cls.glutathione)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(carbonyls))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, carbonyls, cls.glutathione, product
            )

        return final_substrates

    @classmethod
    def gluthationylation_alkyl_halides(cls, comp: Mol) -> list[Mol]:
        """Perform gluthationylation of alkyl halides"""
        alkyl_halides = "[#9,#17,#35,#53,#85,#117:1]-[#6:2]"
        product = "[#6:2]-[#16]-[#6](-[#7]-[#6](=[#8])-[#6]-[#6]-[#6](-[#7+])-[#6](=[#8])-[#8-])-[#6](=[#8])-[#7]-[#6]-[#6](=[#8])-[#8].[#9,#17,#35,#53,#85,#117:1]"
        react2 = Chem.MolFromSmarts(cls.glutathione)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(alkyl_halides))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, alkyl_halides, cls.glutathione, product
            )

        return final_substrates

    @classmethod
    def gluthationylation_metal_halides(cls, comp: Mol) -> list[Mol]:
        """Perform gluthationylation of metal halides"""
        metal_halides = "[#9,#17,#35,#53,#85,#117:1]-[]-[#9,#17,#35,#53,#85,#117:3]"
        product = ""
        react2 = Chem.MolFromSmarts(cls.glutathione)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(metal_halides))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, metal_halides, cls.glutathione, product
            )

        return final_substrates

    @classmethod
    def gluthationylation_quinone_imines(cls, comp: Mol) -> list[Mol]:
        """Perform gluthationylation of quinone imines"""
        quinone_imines = "[#7:2]=[#6:3]:1:[#6:4]:[#6:5]:[#6:6](=[#8]):[#6:7]:[#6:8]:1"
        product = "[#7:2]-[#6:1]:1:[#6:2]:[#6:3]:[#6:4](=[#8]):[#6:5]:[#6:6](-[#16]-[#6](-[#7]-[#6](=[#8])-[#6]-[#6]-[#6](-[#7+])-[#6](=[#8])-[#8+])-[#6](=[#8])-[#7]-[#6]-[#6](=[#8])-[#8]):1"
        react2 = Chem.MolFromSmarts(cls.glutathione)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(quinone_imines))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, quinone_imines, cls.glutathione, product
            )

        return final_substrates

    # Methylation
    @classmethod
    def methylation_catechols(cls, comp: Mol) -> list[Mol]:
        """Perform methylation of catechols"""
        catechols = "[#8]-[#6:1]:1:[#6:2](-[#8:7]):[#6:3]:[#6:4]:[#6:5]:[#6:6]:1"
        product = "[#8]-[#6:1]:1:[#6:2](-[#8:7]-[#6]):[#6:3]:[#6:4]:[#6:5]:[#6:6]:1"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(catechols))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, catechols, cls.methyl_group, product
            )

        return final_substrates

    @classmethod
    def methylation_hydroxyindoles(cls, comp: Mol) -> list[Mol]:
        """Perform methylation of hydroxyindoles"""
        hydroxyindoles = "[#8]-[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:2:[#6:7]:[#6:8]:[#7:9]:[#6:5]:2:[#6:6]:1"
        product = "[#6]-[#8:10]-[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:2:[#6:7]:[#6:8]:[#7:9]:[#6:5]:2:[#6:6]:1"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(hydroxyindoles))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, hydroxyindoles, cls.methyl_group, product
            )

        return final_substrates

    @classmethod
    def methylation_amines(cls, comp: Mol) -> list[Mol]:
        """Perform methylation of primary and secondary amines"""
        matches_1 = comp.GetSubstructMatches(Chem.MolFromSmarts("[#7]-[#6]"))
        matches_2 = comp.GetSubstructMatches(Chem.MolFromSmarts("[#7](-[#6])(-[#6])"))

        final_substrates = []

        if len(matches_2) > 0:
            amine = "[#7:1](-[#6:2])-[#6:3]"
            product = "[#6]-[#7:1](-[#6:2])-[#6:3]"
            final_substrates = cls._perform_reaction(
                comp, cls.glucuronic_acid_mol, amine, cls.glucuronic_acid, product
            )

        elif len(matches_1) > 0:
            amine = "[#7:1]-[#6:3]"
            product = "[#6]-[#7:1](-[#6])-[#6:3]"
            final_substrates = cls._perform_reaction(
                comp, cls.glucuronic_acid_mol, amine, cls.glucuronic_acid, product
            )

        return final_substrates

    @classmethod
    def methylation_pyridines(cls, comp: Mol) -> list[Mol]:
        """Perform methylation of pyridines"""
        pyridines = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[#7]:[#6:6]:1"
        product = "[#6:1]:1:[#6:2]:[#6:3]:[#6:4]:[#7+](-[#6]):[#6:6]:1"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(pyridines))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, pyridines, cls.methyl_group, product
            )

        return final_substrates

    @classmethod
    def methylation_imidazoles(cls, comp: Mol) -> list[Mol]:
        """Perform methylation imidazoles"""
        imidazoles = "[#6:1]:1:[#6:2]:[#7:3]:[#6:4]:[#7:5]:[#6:6]:1"
        product = "[#6:1]:1:[#6:2]:[#7:3]:[#6:4]:[#7:5](-[#6]):[#6:6]:1"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(imidazoles))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, imidazoles, cls.methyl_group, product
            )

        return final_substrates

    @classmethod
    def methylation_thiols(cls, comp: Mol) -> list[Mol]:
        """Perform methylation thiols"""
        thiols = "[#6:1]-[#16:2]"
        product = "[#6:1]-[#16:2]-[#6]"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(thiols))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, thiols, cls.methyl_group, product
            )

        return final_substrates

    @classmethod
    def methylation_thiopurines(cls, comp: Mol) -> list[Mol]:
        """Perform methylation thiopurines"""
        thiopurines = "[#7:1]:1:[#6:2]:[#7:3]:[#6:4]:2:[#7:7]:[#6:8]:[#7:9]:[#6:5]:2:[#6:6](-[#16:10]):1"
        product = "[#7:1]:1:[#6:2]:[#7:3]:[#6:4]:2:[#7:7]:[#6:8]:[#7:9]:[#6:5]:2:[#6:6](-[#16:10]-[#6]):1"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(thiopurines))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, thiopurines, cls.methyl_group, product
            )

        return final_substrates

    @classmethod
    def methylation_arsen(cls, comp: Mol) -> list[Mol]:
        """Perform methylation of arsen"""
        arsen = "[#8:1]-[#33:2](-[#8:3])-[#8:4]"
        product = "[#8:1]=[#33:2](-[#8:3])(-[#6])-[#6]"
        react2 = Chem.MolFromSmarts(cls.methyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(arsen))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, arsen, cls.methyl_group, product
            )

        return final_substrates

    # Acetylation
    @classmethod
    def acetylation_first_amines(cls, comp: Mol) -> list[Mol]:
        """Perform acetylation of first amines"""
        amine = "[#6:1]-[#7:2]"
        product = "[#6:1]-[#7:2]-[#6](=[#8])-[#6]"
        react2 = Chem.MolFromSmarts(cls.acetyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(amine))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, amine, cls.acetyl_group, product
            )

        return final_substrates

    @classmethod
    def acetylation_hydrazine(cls, comp: Mol) -> list[Mol]:
        """Perform acetylation of hydrazine"""
        hydrazine = "[#6:1]-[#7:2]-[#7:3]"
        product = "[#6:1]-[#7:2]-[#7:3]-[#6](=[#8])-[#6]"
        react2 = Chem.MolFromSmarts(cls.acetyl_group)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(hydrazine))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, hydrazine, cls.acetyl_group, product
            )

        return final_substrates

    # Amino Acid Conjugation
    @classmethod
    def conjugation_glycine(cls, comp: Mol) -> list[Mol]:
        """Perform conjugation of glycine"""
        acetyl = "[#6:1]-[#6](=[#8])-[#8:2]"
        glycine = "[#7]-[#6]-[#6](=[#8])-[#8]"
        product = "[#6:1]-[#6](=[#8])-[#7]-[#6]-[#6](=[#8])-[#8]"
        react2 = Chem.MolFromSmarts(glycine)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(acetyl))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, acetyl, glycine, product
            )

        return final_substrates

    @classmethod
    def conjugation_glutamine(cls, comp: Mol) -> list[Mol]:
        """Perform conjugation of glutamine"""
        acetyl = "[#6:1]-[#6](=[#8])-[#8:2]"
        glutamine = "[#7]-[#6](-[#6]-[#6]-[#6](=[#8])-[#7])-[#6](=[#8])-[#8]"
        product = (
            "[#6:1]-[#6](=[#8])-[#7]-[#6](-[#6]-[#6]-[#6](=[#8])-[#7])-[#6](=[#8])-[#8]"
        )
        react2 = Chem.MolFromSmarts(glutamine)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(acetyl))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, acetyl, glutamine, product
            )

        return final_substrates

    @classmethod
    def conjugation_taurine(cls, comp: Mol) -> list[Mol]:
        """Perform conjugation of taurine"""
        acetyl = "[#6:1]-[#6](=[#8])-[#8:2]"
        taurine = "[#7]-[#6]-[#6]-[#16](=[#8])(=[#8])-[#8-]"
        product = "[#6:1]-[#6](=[#8])-[#7]-[#6]-[#6]-[#16](=[#8])(=[#8])-[#8-]"
        react2 = Chem.MolFromSmarts(taurine)

        matches = comp.GetSubstructMatches(Chem.MolFromSmarts(acetyl))
        final_substrates = []

        if len(matches) > 0:
            final_substrates = cls._perform_reaction(
                comp, react2, acetyl, taurine, product
            )

        return final_substrates
