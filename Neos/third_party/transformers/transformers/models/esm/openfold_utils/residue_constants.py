# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in AlphaFold."""

import collections
import copy
import functools
from collections.abc import Mapping, Sequence
from importlib import resources

import numpy as np






ca_ca = 3.80209737096




chi_angles_atoms: dict[str, list[list[str]]] = {
    "ALA": [],
    
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}



chi_angles_mask: list[list[float]] = [
    [0.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 1.0, 1.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 1.0, 0.0],  
    [1.0, 1.0, 1.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 1.0, 1.0],  
    [1.0, 1.0, 1.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
]



chi_pi_periodic: list[list[float]] = [
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 1.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 1.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 1.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 1.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
]













rigid_group_atom_positions: dict[str, list[tuple[str, int, tuple[float, float, float]]]] = {
    "ALA": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
    ],
    "ARG": [
        ("N", 0, (-0.524, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, -0.000)),
        ("CB", 0, (-0.524, -0.778, -1.209)),
        ("O", 3, (0.626, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, -0.000)),
        ("CD", 5, (0.564, 1.414, 0.000)),
        ("NE", 6, (0.539, 1.357, -0.000)),
        ("NH1", 7, (0.206, 2.301, 0.000)),
        ("NH2", 7, (2.078, 0.978, -0.000)),
        ("CZ", 7, (0.758, 1.093, -0.000)),
    ],
    "ASN": [
        ("N", 0, (-0.536, 1.357, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.531, -0.787, -1.200)),
        ("O", 3, (0.625, 1.062, 0.000)),
        ("CG", 4, (0.584, 1.399, 0.000)),
        ("ND2", 5, (0.593, -1.188, 0.001)),
        ("OD1", 5, (0.633, 1.059, 0.000)),
    ],
    "ASP": [
        ("N", 0, (-0.525, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, 0.000, -0.000)),
        ("CB", 0, (-0.526, -0.778, -1.208)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("CG", 4, (0.593, 1.398, -0.000)),
        ("OD1", 5, (0.610, 1.091, 0.000)),
        ("OD2", 5, (0.592, -1.101, -0.003)),
    ],
    "CYS": [
        ("N", 0, (-0.522, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.524, 0.000, 0.000)),
        ("CB", 0, (-0.519, -0.773, -1.212)),
        ("O", 3, (0.625, 1.062, -0.000)),
        ("SG", 4, (0.728, 1.653, 0.000)),
    ],
    "GLN": [
        ("N", 0, (-0.526, 1.361, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.525, -0.779, -1.207)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("CG", 4, (0.615, 1.393, 0.000)),
        ("CD", 5, (0.587, 1.399, -0.000)),
        ("NE2", 6, (0.593, -1.189, -0.001)),
        ("OE1", 6, (0.634, 1.060, 0.000)),
    ],
    "GLU": [
        ("N", 0, (-0.528, 1.361, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.526, -0.781, -1.207)),
        ("O", 3, (0.626, 1.062, 0.000)),
        ("CG", 4, (0.615, 1.392, 0.000)),
        ("CD", 5, (0.600, 1.397, 0.000)),
        ("OE1", 6, (0.607, 1.095, -0.000)),
        ("OE2", 6, (0.589, -1.104, -0.001)),
    ],
    "GLY": [
        ("N", 0, (-0.572, 1.337, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.517, -0.000, -0.000)),
        ("O", 3, (0.626, 1.062, -0.000)),
    ],
    "HIS": [
        ("N", 0, (-0.527, 1.360, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, 0.000, 0.000)),
        ("CB", 0, (-0.525, -0.778, -1.208)),
        ("O", 3, (0.625, 1.063, 0.000)),
        ("CG", 4, (0.600, 1.370, -0.000)),
        ("CD2", 5, (0.889, -1.021, 0.003)),
        ("ND1", 5, (0.744, 1.160, -0.000)),
        ("CE1", 5, (2.030, 0.851, 0.002)),
        ("NE2", 5, (2.145, -0.466, 0.004)),
    ],
    "ILE": [
        ("N", 0, (-0.493, 1.373, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, -0.000, -0.000)),
        ("CB", 0, (-0.536, -0.793, -1.213)),
        ("O", 3, (0.627, 1.062, -0.000)),
        ("CG1", 4, (0.534, 1.437, -0.000)),
        ("CG2", 4, (0.540, -0.785, -1.199)),
        ("CD1", 5, (0.619, 1.391, 0.000)),
    ],
    "LEU": [
        ("N", 0, (-0.520, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, -0.000)),
        ("CB", 0, (-0.522, -0.773, -1.214)),
        ("O", 3, (0.625, 1.063, -0.000)),
        ("CG", 4, (0.678, 1.371, 0.000)),
        ("CD1", 5, (0.530, 1.430, -0.000)),
        ("CD2", 5, (0.535, -0.774, 1.200)),
    ],
    "LYS": [
        ("N", 0, (-0.526, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.524, -0.778, -1.208)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("CG", 4, (0.619, 1.390, 0.000)),
        ("CD", 5, (0.559, 1.417, 0.000)),
        ("CE", 6, (0.560, 1.416, 0.000)),
        ("NZ", 7, (0.554, 1.387, 0.000)),
    ],
    "MET": [
        ("N", 0, (-0.521, 1.364, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, 0.000, 0.000)),
        ("CB", 0, (-0.523, -0.776, -1.210)),
        ("O", 3, (0.625, 1.062, -0.000)),
        ("CG", 4, (0.613, 1.391, -0.000)),
        ("SD", 5, (0.703, 1.695, 0.000)),
        ("CE", 6, (0.320, 1.786, -0.000)),
    ],
    "PHE": [
        ("N", 0, (-0.518, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.524, 0.000, -0.000)),
        ("CB", 0, (-0.525, -0.776, -1.212)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("CG", 4, (0.607, 1.377, 0.000)),
        ("CD1", 5, (0.709, 1.195, -0.000)),
        ("CD2", 5, (0.706, -1.196, 0.000)),
        ("CE1", 5, (2.102, 1.198, -0.000)),
        ("CE2", 5, (2.098, -1.201, -0.000)),
        ("CZ", 5, (2.794, -0.003, -0.001)),
    ],
    "PRO": [
        ("N", 0, (-0.566, 1.351, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, -0.000, 0.000)),
        ("CB", 0, (-0.546, -0.611, -1.293)),
        ("O", 3, (0.621, 1.066, 0.000)),
        ("CG", 4, (0.382, 1.445, 0.0)),
        
        ("CD", 5, (0.477, 1.424, 0.0)),  
    ],
    "SER": [
        ("N", 0, (-0.529, 1.360, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, -0.000)),
        ("CB", 0, (-0.518, -0.777, -1.211)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("OG", 4, (0.503, 1.325, 0.000)),
    ],
    "THR": [
        ("N", 0, (-0.517, 1.364, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, -0.000)),
        ("CB", 0, (-0.516, -0.793, -1.215)),
        ("O", 3, (0.626, 1.062, 0.000)),
        ("CG2", 4, (0.550, -0.718, -1.228)),
        ("OG1", 4, (0.472, 1.353, 0.000)),
    ],
    "TRP": [
        ("N", 0, (-0.521, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, 0.000)),
        ("CB", 0, (-0.523, -0.776, -1.212)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.609, 1.370, -0.000)),
        ("CD1", 5, (0.824, 1.091, 0.000)),
        ("CD2", 5, (0.854, -1.148, -0.005)),
        ("CE2", 5, (2.186, -0.678, -0.007)),
        ("CE3", 5, (0.622, -2.530, -0.007)),
        ("NE1", 5, (2.140, 0.690, -0.004)),
        ("CH2", 5, (3.028, -2.890, -0.013)),
        ("CZ2", 5, (3.283, -1.543, -0.011)),
        ("CZ3", 5, (1.715, -3.389, -0.011)),
    ],
    "TYR": [
        ("N", 0, (-0.522, 1.362, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.524, -0.000, -0.000)),
        ("CB", 0, (-0.522, -0.776, -1.213)),
        ("O", 3, (0.627, 1.062, -0.000)),
        ("CG", 4, (0.607, 1.382, -0.000)),
        ("CD1", 5, (0.716, 1.195, -0.000)),
        ("CD2", 5, (0.713, -1.194, -0.001)),
        ("CE1", 5, (2.107, 1.200, -0.002)),
        ("CE2", 5, (2.104, -1.201, -0.003)),
        ("OH", 5, (4.168, -0.002, -0.005)),
        ("CZ", 5, (2.791, -0.001, -0.003)),
    ],
    "VAL": [
        ("N", 0, (-0.494, 1.373, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, -0.000, -0.000)),
        ("CB", 0, (-0.533, -0.795, -1.213)),
        ("O", 3, (0.627, 1.062, -0.000)),
        ("CG1", 4, (0.540, 1.429, -0.000)),
        ("CG2", 4, (0.533, -0.776, 1.203)),
    ],
}


residue_atoms: dict[str, list[str]] = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}








residue_atom_renaming_swaps: dict[str, dict[str, str]] = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}


van_der_waals_radius: dict[str, float] = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

Bond = collections.namedtuple("Bond", ["atom1_name", "atom2_name", "length", "stddev"])
BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)


def map_structure_with_atom_order(in_list: list, first_call: bool = True) -> list:
    
    if first_call:
        in_list = copy.deepcopy(in_list)
    for i in range(len(in_list)):
        if isinstance(in_list[i], list):
            in_list[i] = map_structure_with_atom_order(in_list[i], first_call=False)
        elif isinstance(in_list[i], str):
            in_list[i] = atom_order[in_list[i]]
        else:
            raise TypeError("Unexpected type when mapping nested lists!")
    return in_list


@functools.cache
def load_stereo_chemical_props() -> tuple[
    Mapping[str, list[Bond]],
    Mapping[str, list[Bond]],
    Mapping[str, list[BondAngle]],
]:
    """Load stereo_chemical_props.txt into a nice structure.

    Load literature values for bond lengths and bond angles and translate bond angles into the length of the opposite
    edge of the triangle ("residue_virtual_bonds").

    Returns:
      residue_bonds: dict that maps resname --> list of Bond tuples residue_virtual_bonds: dict that maps resname -->
      list of Bond tuples residue_bond_angles: dict that maps resname --> list of BondAngle tuples
    """
    
    stereo_chemical_props = resources.read_text("openfold.resources", "stereo_chemical_props.txt")

    lines_iter = iter(stereo_chemical_props.splitlines())
    
    residue_bonds: dict[str, list[Bond]] = {}
    next(lines_iter)  
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, bond_length, stddev = line.split()
        atom1, atom2 = bond.split("-")
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(Bond(atom1, atom2, float(bond_length), float(stddev)))
    residue_bonds["UNK"] = []

    
    residue_bond_angles: dict[str, list[BondAngle]] = {}
    next(lines_iter)  
    next(lines_iter)  
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, angle_degree, stddev_degree = line.split()
        atom1, atom2, atom3 = bond.split("-")
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180.0 * np.pi,
                float(stddev_degree) / 180.0 * np.pi,
            )
        )
    residue_bond_angles["UNK"] = []

    def make_bond_key(atom1_name: str, atom2_name: str) -> str:
        """Unique key to lookup bonds."""
        return "-".join(sorted([atom1_name, atom2_name]))

    
    residue_virtual_bonds: dict[str, list[Bond]] = {}
    for resname, bond_angles in residue_bond_angles.items():
        
        bond_cache: dict[str, Bond] = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

            
            
            gamma = ba.angle_rad
            length = np.sqrt(bond1.length**2 + bond2.length**2 - 2 * bond1.length * bond2.length * np.cos(gamma))

            
            dl_outer = 0.5 / length
            dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2 + (dl_db1 * bond1.stddev) ** 2 + (dl_db2 * bond2.stddev) ** 2
            )
            residue_virtual_bonds[resname].append(Bond(ba.atom1_name, ba.atom3name, length, stddev))

    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)




between_res_bond_length_c_n: tuple[float, float] = (1.329, 1.341)
between_res_bond_length_stddev_c_n: tuple[float, float] = (0.014, 0.016)


between_res_cos_angles_c_n_ca: tuple[float, float] = (-0.5203, 0.0353)  
between_res_cos_angles_ca_c_n: tuple[float, float] = (-0.4473, 0.0311)  



atom_types: list[str] = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order: dict[str, int] = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  



restype_name_to_atom14_names: dict[str, list[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}





restypes: list[str] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order: dict[str, int] = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  
unk_restype_index = restype_num  

restypes_with_x: list[str] = restypes + ["X"]
restype_order_with_x: dict[str, int] = {restype: i for i, restype in enumerate(restypes_with_x)}


def sequence_to_onehot(sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain amino acid 'X', an error will be thrown.
        If False, any amino acid not in the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 without any gaps. Got: %s"
            % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping["X"])
            else:
                raise ValueError(f"Invalid character in the sequence: {aa_type}")
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr


restype_1to3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}






restype_3to1: dict[str, str] = {v: k for k, v in restype_1to3.items()}


unk_restype = "UNK"

resnames: list[str] = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx: dict[str, int] = {resname: i for i, resname in enumerate(resnames)}









HHBLITS_AA_TO_ID: dict[str, int] = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
    "-": 21,
}


ID_TO_HHBLITS_AA: dict[int, str] = {
    0: "A",
    1: "C",  
    2: "D",  
    3: "E",  
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",  
    21: "-",
}

restypes_with_x_and_gap: list[str] = restypes + ["X", "-"]
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE: tuple[int, ...] = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i]) for i in range(len(restypes_with_x_and_gap))
)


def _make_standard_atom_mask() -> np.ndarray:
    """Returns [num_res_types, num_atom_types] mask array."""
    
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()




def chi_angle_atom(atom_index: int) -> np.ndarray:
    """Define chi-angle rigid groups via one-hot representations."""
    chi_angles_index = {}
    one_hots = []

    for k, v in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1] * (4 - len(indices)))
        chi_angles_index[k] = indices

    for r in restypes:
        res3 = restype_1to3[r]
        one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
        one_hots.append(one_hot)

    one_hots.append(np.zeros([4, atom_type_num]))  
    one_hot = np.stack(one_hots, axis=0)
    one_hot = np.transpose(one_hot, [0, 2, 1])

    return one_hot


chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)


chi_angles_atom_indices_list: list[list[list[str]]] = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices_ours: list = map_structure_with_atom_order(chi_angles_atom_indices_list)
chi_angles_atom_indices = np.array(
    [chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms))) for chi_atoms in chi_angles_atom_indices_list]
)



chi_groups_for_atom: dict[tuple[str, str], list[tuple[int, int]]] = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
    for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
        for atom_i, atom in enumerate(chi_group):
            chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)


def _make_rigid_transformation_4x4(ex: np.ndarray, ey: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    
    ex_normalized = ex / np.linalg.norm(ex)

    
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m






restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=int)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)


def _make_rigid_group_constants() -> None:
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            restype_atom37_mask[restype, atomtype] = 1
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            restype_atom14_mask[restype, atom14idx] = 1
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position

    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_positions: dict[str, np.ndarray] = {
            name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]
        }

        
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

        
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["N"],
        )
        restype_rigid_group_default_frame[restype, 2, :, :] = mat

        
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],
            translation=atom_positions["C"],
        )
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            restype_rigid_group_default_frame[restype, 4, :, :] = mat

        
        
        
        
        
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


def make_atom14_dists_bounds(
    overlap_tolerance: float = 1.5,
    bond_length_tolerance_factor: int = 15,
) -> dict[str, np.ndarray]:
    """compute upper and lower bounds for bonds to assess violations."""
    restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_list = restype_name_to_atom14_names[resname]

        
        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if (not atom2_name) or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 1e10
                restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

        
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {
        "lower_bound": restype_atom14_bond_lower_bound,  
        "upper_bound": restype_atom14_bond_upper_bound,  
        "stddev": restype_atom14_bond_stddev,  
    }


restype_atom14_ambiguous_atoms = np.zeros((21, 14), dtype=np.float32)
restype_atom14_ambiguous_atoms_swap_idx: np.ndarray = np.tile(np.arange(14, dtype=int), (21, 1))


def _make_atom14_ambiguity_feats() -> None:
    for res, pairs in residue_atom_renaming_swaps.items():
        res_idx = restype_order[restype_3to1[res]]
        for atom1, atom2 in pairs.items():
            atom1_idx = restype_name_to_atom14_names[res].index(atom1)
            atom2_idx = restype_name_to_atom14_names[res].index(atom2)
            restype_atom14_ambiguous_atoms[res_idx, atom1_idx] = 1
            restype_atom14_ambiguous_atoms[res_idx, atom2_idx] = 1
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom1_idx] = atom2_idx
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom2_idx] = atom1_idx


_make_atom14_ambiguity_feats()


def aatype_to_str_sequence(aatype: Sequence[int]) -> str:
    return "".join([restypes_with_x[aatype[i]] for i in range(len(aatype))])
