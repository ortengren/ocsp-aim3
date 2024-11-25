import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

from ase import Atoms, geometry, neighborlist
from ase.io import read, write
from ase.neighborlist import NeighborList
from ase.visualize import view

from scipy import sparse
from scipy.spatial.transform import Rotation
from scipy import linalg

import networkx as nx

import metatensor
from featomic import SoapPowerSpectrum
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys


def build_graph(mol: Atoms, nl=None):
    if nl == None:
        nl = neighborlist.build_neighbor_list(mol, self_interaction=False, bothways=True)
    G = nx.Graph()
    for i in range(len(mol)):
        atom = mol[i]
        nb_indices, offsets = nl.get_neighbors(i)
        nb_atoms = [mol[a] for a in nb_indices]
        el = [(i, nb) for nb in nb_indices]
        G.add_edges_from(el)
        G.nodes[i]["atom"] = mol[i]
    return G


def get_rings(frame, graph=None):
    if graph == None:
        G = build_graph(frame)
    else:
        G = graph
    rings = nx.cycle_basis(G)
    return rings


def get_cluster_data(ring):
    moments, axes = ring.get_moments_of_inertia(vectors=True)
    mass = np.sum([atom.mass for atom in ring])
    Ix, Iy, Iz = moments
    a2 = ((5 * (Iz + Iy - Ix)) / 2) / mass
    a = np.sqrt(a2)
    b2 = (((5 * Iz) - (5 * (Iy + Ix))) / 2) / mass
    print(b2)
    b = np.sqrt(b2)
    c2 = (((5 * (Ix + Iy)) - (5 * Iz)) / 2) / mass
    c = np.sqrt(c2)
    return {"axes": [a, b, c], "moments": moments, "eigenvectors": axes, "mass": mass}


def get_ellipsoids(frame, axes):
    coms = []
    quats = []
    positions = []
    dim1 = []
    dim2 = []
    dim3 = []
    for i, ring in enumerate(get_rings(frame)):
        cluster = frame[[a for a in range(len(frame)) if (a in ring and frame.arrays["numbers"][a] == 6)]]
        if len(cluster) != 6:
            continue
        dist_vecs, _ = geometry.get_distances(cluster.positions)
        pos_vecs = cluster.positions[0] + dist_vecs[0]
        com = pos_vecs.mean(axis=0).flatten()
        data = get_cluster_data(cluster)
        rot = np.asarray(data["eigenvectors"]).T
        if np.isclose(np.linalg.det(rot), -1):
            rot = np.matmul(rot, [[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        quat = Rotation.from_matrix(rot).as_quat()
        quat = np.roll(quat, 1)
        quats.append(quat)
        positions.append(pos_vecs)
        coms.append(com)
        print(coms)
        dim1.append(axes[0])
        dim2.append(axes[1])
        dim3.append(axes[2])
    if len(coms) == 0:
        return None
    ell_frame = Atoms(positions=np.vstack(coms), cell=frame.cell, pbc=frame.pbc)
    ell_frame.arrays["quaternions"] = np.vstack(quats)
    ell_frame.arrays["c_diameter[1]"] = np.array(dim1).flatten()
    ell_frame.arrays["c_diameter[2]"] = np.array(dim2).flatten()
    ell_frame.arrays["c_diameter[3]"] = np.array(dim3).flatten()
    return ell_frame


def get_ell_frames(frames, axes):
    ell_frames = []
    failed_frames = []
    for i, frame in enumerate(frames):
        result = get_ellipsoids(frame, axes)
        if result is None:
            failed_frames.append(i)
            continue
        ell_frames.append(result)
    return ell_frames, failed_frames


mols = read("benzene_containing_molecules.xyz", ":")

NLIST_KWARGS = {
    "skin": 0.3,   # doesn't matter for this application.
    "sorted": False,
    "self_interaction": False,
    "bothways": True
}

ell_frames, failed = [get_ellipsoids(frame, [2.5, 2.5, 0.5]) for frame in mols]
#%%
print(len(failed))