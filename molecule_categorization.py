import ase
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from ase.visualize import view
from ase.visualize.ngl import view_ngl
from ase.io import read, write


class MoleculeData:
    def __init__(self, str):
        self.lines = str.split("\n")
        lines = self.lines
        self.num_atoms = lines[0]
        self.lattice: np.array = np.array(lines[1][0:lines[1].index(" Properties")].split('"')[1].split())
        props = lines[1][lines[1].index("Properties") + 11:-1]
        self.dft_energy_ryd = props[props.index("dft_energy_ryd=") + 16:props.index(" molecule_idx")]
        self.molecule_idx = props[props.index("molecule_idx") + 13:props.index(" crystal_idx")]
        self.crystal_idx = props[props.index("crystal_idx") + 12:props.index(" motif_idx")]
        m_idx = props[props.index("motif_idx") + 17:props.index("motif_names")]
        m_idx = m_idx[2:].split("[")
        l = []
        for el in m_idx:
            sub = [int(c) for c in el if c.isdecimal()]
            l.append(sub)
        self.motif_idx = l
        m_names = props[props.index("motif_names") + 19:props.index(" pbc")]
        m_names = m_names.split()
        self.motif_names = [name.strip("\\[,\"]") for name in m_names]
        self.pbc = lines[1][lines[1].index("pbc=") + 5:].rstrip('"')
        self.atoms = lines[2:]
        xyz = ""
        for line in lines[2:]:
            xyz += line + "\n"
        self.xyz = f"{self.num_atoms}\n" + xyz[:-1]


def read_tagged_xyz(tagged_filename: str, untagged_filename: str) -> list[ase.Atoms]:
    with open(tagged_filename, "r") as f:
        lines = f.readlines()
    blocks = []
    block = lines[0]
    i = 1
    while i < len(lines):
        if lines[i][0].isdecimal():
            blocks.append(block)
            block = ""
        block = block + lines[i]
        i += 1
    ase_mols = read(untagged_filename, index=":")
    mols = [MoleculeData(block) for block in blocks]
    mols = list(zip(mols, ase_mols))
    frames = []
    for data, frame in mols:
        frame.info["lattice"] = data.lattice
        frame.info["dft_energy_ryd"] = data.dft_energy_ryd
        frame.arrays["motif_idx"] = np.array(data.motif_idx, dtype=object)
        frame.arrays["motif_names"] = np.array(data.motif_names)
        frames.append(frame)
    return frames


frames = read_tagged_xyz("data/all_relaxed_molecules_tagged.xyz", "data/all_relaxed_molecules.xyz")
bframes_idx = [i for i, frame in enumerate(frames) if "benzene" in frame.arrays["motif_names"]]
bframes = read("data/all_relaxed_molecules.xyz", index=":")
bframes = [frame for i, frame in enumerate(bframes) if i in bframes_idx]
#bframes = [frame for frame in frames if "benzene" in frame.arrays["motif_names"]]

write("benzene_containing_molecules.xyz", bframes)
