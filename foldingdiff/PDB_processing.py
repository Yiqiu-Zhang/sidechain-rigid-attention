import torch
from Bio import PDB
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
AA_TO_ID = {
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
    "Y": 19,
    "Z": 3,

}

ANGLES = ["X1", "X2", "X3","X4"]
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [], #[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]] PRO chi angle was replaced to 0, 为了把PRO看作一个刚体
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}
restype_1to3 = {
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
restype_3to1 = {v: k for k, v in restype_1to3.items()}
bb_atoms = ['N', 'CA', 'C', 'O']

restype_name_to_rigid_idx = {
    "ALA": [2],
    "ARG": [1,8,8,8,19],
    "ASN": [1,8,16],
    "ASP": [1,8,15],
    "CYS": [1,6],
    "GLN": [1,8,8,16],
    "GLU": [1,8,8,15],
    "GLY": [1],
    "HIS": [1,8,17],
    "ILE": [1,9,10],
    "LEU": [1,8,7],
    "LYS": [1,8,8,8,18],
    "MET": [1,8,8,11],
    "PHE": [1,8,12],
    "PRO": [3],
    "SER": [1,4],
    "THR": [1,5],
    "TRP": [1,8,14],
    "TYR": [1,8,13],
    "VAL": [1,7],
}
rigid_type_property = [
    [1,0,0,0,0,0],
    [1,0,0,0,0,0],
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,1,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [0,0,0,0,1,0],
    [0,1,0,0,0,0],
    [0,0,0,1,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
]

def acid_to_number(seq, mapping):
    num_list = []
    for acid in seq:
        if acid in mapping:
            num_list.append(mapping[acid])
    return num_list

def get_torsion_seq(pdb_path):
    torsion_list = []
    chi_mask =[]
    seq = []
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('name', pdb_path)
    model = structure[0]
    chain = model.child_list[0]
    X = []
    L = len(chain)
    rigid_type = np.zeros((L,5))
    rigid_type_mask = np.zeros((L,5))
    rigid_property = np.zeros((L,5,6))

    for res_idx, res in enumerate(chain):

        chi_list = [0] * 4
        temp_mask =[0] * 4
        # Skip hetero atoms
        if res.id[0] != " ":
            continue

        res_name = res.resname
        seq.append(restype_3to1[res_name])
        res_torsion_atom_list = chi_angles_atoms[res_name]
        res_rigid_group_list = restype_name_to_rigid_idx[res_name]
        X.append([res[a].get_coord() for a in bb_atoms])

        for i, rigid in enumerate(res_rigid_group_list):
            rigid_type[res_idx][i] = rigid
            rigid_type_mask[res_idx][i] = 1
            rigid_property[res_idx][i][:] = rigid_type_property[rigid]

        for i, torsion_atoms in enumerate(res_torsion_atom_list):
            vec_atoms_coord = [res[a].get_vector() for a in torsion_atoms]
            angle = PDB.calc_dihedral(*vec_atoms_coord)
            chi_list[i] = angle
            temp_mask[i] = 1
        torsion_list.append(chi_list)
        chi_mask.append(temp_mask)
    chi_mask = np.array(chi_mask)
    torsion_list = np.array(torsion_list)
    X = np.array(X)
    
    #=========
    seq_single = np.array(seq, dtype=np.str)
    
    num_acid_seq = acid_to_number(seq_single, AA_TO_ID)
    num_acid_seq = np.array(num_acid_seq)
    
    X1 = torsion_list[:,0]
    X2 = torsion_list[:,1]
    X3 = torsion_list[:,2]
    X4 = torsion_list[:,3]
    calc_angles = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
    angle_list = pd.DataFrame({k: calc_angles[k].squeeze() for k in ANGLES})

    rigid_type = torch.tensor(rigid_type)
    rigid_type_onehot = F.one_hot(rigid_type,20)  # with the empty rigid type 0
    rigid_type_onehot = rigid_type_onehot * torch.unsqueeze(torch.tensor(rigid_type_mask), -1)

    rigid_type_onehot = np.array(rigid_type_onehot) #(L,5,20)

    dict_struct = {'angles': angle_list,
                   'coords': X,
                   'seq': num_acid_seq,
                   "seq_temp": seq_single,
                   "chi_mask": chi_mask,
                   'rigid_type_onehot': rigid_type_onehot, #(L,5,20)
                   'rigid_property': rigid_property, # (L,5,6)
                   'fname': pdb_path,
                   }
    return dict_struct

#t = get_torsion_seq('./data/1CRN.pdb')
#l = len(t1)
#l2 = len(t['seq'])
#seq= "".join(t["seq"])
#print(seq)  chi_1