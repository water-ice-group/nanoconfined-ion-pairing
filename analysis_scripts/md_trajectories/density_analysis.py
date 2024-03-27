import numpy as np
import MDAnalysis as mda
import glob
import os
import pickle


def get_positions(path, u, atoms, rerun_pos=False):
    """
    Get the positions of atoms from a given MDAnalysis universe.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        u (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        atoms (MDAnalysis.AtomGroup): AtomGroup representing the atoms of interest.
        rerun_pos (bool, optional): If True, recompute the positions even if the output file exists. Default is False.

    Returns:
        numpy.ndarray: Array containing the positions of atoms over the trajectory.
    """
    atom_name = atoms.names[0]
    filename = path + f"positions_{atom_name}.npy"
    if rerun_pos == True or not os.path.exists(filename):
        time = 0
        atom_positions = np.zeros((u.trajectory.n_frames, len(atoms), 3))
        for ts in u.trajectory:
            atom_positions[time, :, :] = atoms.positions
            time += 1
        np.save(filename, np.array(atom_positions))
    else:
        atom_positions = np.load(filename, allow_pickle=True)
    return atom_positions


def compute_density_profiles(path, u, atoms, bins, rerun_pos):
    """
    Compute the density profiles along the z-axis.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        u (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        atoms (MDAnalysis.AtomGroup): AtomGroup representing the atoms of interest.
        bins (numpy.ndarray): Array defining the bins for histogram calculation.
        rerun_pos (bool): If True, recompute the positions even if the output file exists.

    Returns:
        tuple: A tuple containing the positions along the z-axis and the corresponding densities.
    """

    # get arrays of z-positions
    atom_positions = get_positions(path, u, atoms, rerun_pos)[:, :, 2]  # get z-position
    dens, edges = np.histogram(atom_positions, bins, density=True)

    # convert from probability density to number density
    area = u.dimensions[0] * u.dimensions[1]
    n_atoms = len(atoms)
    densities = dens * n_atoms / area

    # center plot around center of slit
    edges = edges[:-1]
    x = edges - (edges[-1] - edges[0]) / 2 - edges[0]

    return x, densities


def run(path, run, run_start=0, delta=0.1, rerun=False, rerun_pos=False):
    """
    Run the density profile computation for different atom types.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        delta (float, optional): Size of the bins for density profile calculation. Default is 0.1.
        rerun (bool, optional): If True, recompute the density profiles even if the output file exists. Default is False.
        rerun_pos (bool, optional): If True, recompute the positions even if the output file exists. Default is False.

    Returns:
        dict: A dictionary containing density profiles for different atom types.
    """
    filename = path + f"dens.pkl"
    if rerun == True or not os.path.exists(filename):
        density_all = {}

        atom_types = list(set(run.atoms.names))
        atom_types.remove("C")  # don't want Carbon
        atom_types.remove("H")  # don't want hydrogen

        carbons = run.select_atoms(f"name C")
        bins = np.arange(carbons[0].position[2], carbons[4].position[2], delta)

        for t1 in atom_types:
            # print('Computing density profile for ' + t1, flush=True)
            t1_atoms = run.select_atoms(f"name {t1}")
            density_all[t1] = compute_density_profiles(
                path, run, t1_atoms, bins, rerun_pos
            )

        fn_out = path + "dens.pkl"
        with open(fn_out, "wb") as f_out:
            pickle.dump(density_all, f_out)

    else:
        with open(path + "dens.pkl", "rb") as handle:
            density_all = pickle.load(handle)

    return density_all


def average_replicates(data, symmetrize=True):
    """
    Compute the average and standard deviation of replicated density profiles.

    Parameters:
        data (list): List of dictionaries containing density profiles for different atom types.
        symmetrize (bool, optional): If True, compute error on symmetrized density profiles. Default is True.

    Returns:
        tuple: A tuple containing the averaged density profiles and corresponding errors.
    """

    data_avg = {}
    data_err = {}

    for key in data[0].keys():
        values = [d[key] for d in data]

        if symmetrize == True:  # compute error on symmetrized density profiles
            y_temp = []
            half_length = int(len(values[0][1]) / 2)
            for value in values:
                half_y = (value[1][:half_length] + np.flip(value[1])[:half_length]) / 2
                y = np.concatenate((half_y, np.flip(half_y)))
                y_temp.append(y)
            err_values = np.std(y_temp, axis=0)
            avg_values = np.mean(y_temp, axis=0)

        else:
            err_values = np.std([value[1] for value in values], axis=0)
            avg_values = np.mean([value[1] for value in values], axis=0)

        x = value[0][: len(avg_values)]

        data_avg[key] = [x, avg_values]
        data_err[key] = err_values

    return data_avg, data_err
