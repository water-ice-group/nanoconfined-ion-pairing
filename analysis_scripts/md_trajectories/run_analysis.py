import numpy as np
import MDAnalysis as mda
import glob
import os

import density_analysis
import ion_pairing_analysis as ion_pairing

# Define run info
samplingperiod = 0.5  # ps between data collection
run_start = int(1000 / samplingperiod)  # snapshots to ignore for equilibration
runs = [0, 1, 2, 3, 4]
seps = [6.7, 10.05, 13.4, 16.75]


def create_mda(path, data_file, dcd_file, cat_dcd=False):
    """
    Create a MDAnalysis Universe object for a simulation.

    Parameters:
        path (str): Path to the directory containing data files.
        data_file (str): Name of the data file.
        dcd_file (str): Name pattern of DCD trajectory files.
        cat_dcd (bool, optional): Whether to concatenate multiple DCD files. Defaults to False.

    Returns:
        MDAnalysis Universe: MDAnalysis Universe object.
    """
    if cat_dcd:
        all_dcd_files = glob.glob(path + dcd_file)
        step = []
        for f in all_dcd_files:
            step.append(int(f.replace(".", "_").split("_")[-2]))
        dcd_file = [x for _, x in sorted(zip(step, all_dcd_files))]
    else:
        dcd_file = path + dcd_file

    run = mda.Universe(path + data_file, dcd_file, format="LAMMPS")

    atom_names = {
        "1": "C",
        "2": "O",
        "3": "H",
        "4": "Na",
        "5": "Cl",
    }

    names = []
    for atom in run.atoms:
        names.append(atom_names[atom.type])

    run.add_TopologyAttr("name", names)

    return run


def generate_times(u, run_start, timestep=samplingperiod):
    """
    Generate time array for simulation frames.

    Parameters:
        u (MDAnalysis Universe): MDAnalysis Universe object.
        run_start (int): Index of the starting frame.
        timestep (float, optional): Time step between frames in picoseconds. Defaults to 0.5.

    Returns:
        numpy.ndarray: Array of time values.
    """
    times = []
    current_step = 0
    for ts in u.trajectory[run_start:]:  # omit equilibration time
        times.append(current_step * timestep)
        current_step += 1
    times = np.array(times)
    return times


def run_pmf_analysis(seps, runs, run_start, rerun=False):
    """
    Perform Potential of Mean Force (PMF) analysis.

    Parameters:
        seps (list): List of carbon slit heights.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """

    filename = "data/pmf_avg.npy"
    if rerun == True or not os.path.exists(filename):
        pmfs = np.empty((len(seps), len(runs)), dtype=object)
        pmf_bins = np.empty((len(seps)), dtype=object)
        for j, sep in enumerate(seps):
            for i, run in enumerate(runs):
                path = f"sep{sep}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                pmfs[j, i], pmf_bins[j] = ion_pairing.get_pmf(
                    path, u, sep, run_start=run_start, rerun=rerun
                )
        pmf_avg = [np.mean(pmfs[j, :]) for j in range(len(seps))]
        pmf_err = [np.std(pmfs[j, :]) for j in range(len(seps))]
        np.save("data/pmf_avg.npy", pmf_avg)
        np.save("data/pmf_err.npy", pmf_err)
        np.save("data/pmf_bins.npy", pmf_bins)


def run_density_profiles(seps, runs, run_start, rerun=False):
    """
    Perform density profiles analysis.

    Parameters:
        seps (list): List of carbon slit heights.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/density_profiles.npy"
    if rerun == True or not os.path.exists(filename):
        density_profiles_all = np.empty((len(seps), len(runs)), dtype=object)
        for j, sep in enumerate(seps):
            for i, run in enumerate(runs):
                path = f"sep{sep}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                density_profiles_all[j, i] = density_analysis.run(
                    path,
                    u,
                    run_start=run_start,
                    delta=0.05,
                    rerun=rerun,
                    rerun_pos=rerun,
                )
        density_profiles_avg = np.empty((len(seps)), dtype=object)
        density_profiles_err = np.empty((len(seps)), dtype=object)
        for j, sep in enumerate(seps):
            (
                density_profiles_avg[j],
                density_profiles_err[j],
            ) = density_analysis.average_replicates(density_profiles_all[j, :])
        np.save("data/density_profiles.npy", density_profiles_avg)
        np.save("data/density_profiles_err.npy", density_profiles_err)


def run_ion_pairing_fracs(seps, runs, cip_dists, ssip_dists, run_start, rerun=False):
    """
    Compute ion pairing fractions.

    Parameters:
        seps (list): List of carbon slit heights.
        runs (list): List of replicate run indices.
        cip_dists (numpy.ndarray): Cutoff distances for contact ion pairs.
        ssip_dists (numpy.ndarray): Cutoff distances for solvent-separated ion pairs.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/ion_pairing_fracs.npy"
    if rerun == True or not os.path.exists(filename):
        ion_pairing_all = np.empty((len(seps), len(runs), 3), dtype=float)
        for j, sep in enumerate(seps):
            for i, run in enumerate(runs):
                path = f"sep{sep}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                ion_pairing_all[j, i, :], _ = ion_pairing.compute_ionPairFrac(
                    path,
                    u,
                    cip_dist=cip_dists[j],
                    ssip_dist=ssip_dists[j],
                    run_start=run_start,
                    rerun=rerun,
                )
        ion_pairing_avg = np.empty((len(seps), 3), dtype=float)
        ion_pairing_err = np.empty((len(seps), 3), dtype=float)
        for j, sep in enumerate(seps):
            (
                ion_pairing_avg[j, :],
                ion_pairing_err[j, :],
            ) = ion_pairing.average_replicates(ion_pairing_all[j, :])
        np.save("data/ion_pairing_fracs.npy", ion_pairing_avg)
        np.save("data/ion_pairing_fracs_err.npy", ion_pairing_err)


def run_ion_pair_residence_times(seps, runs, run_start, cip_cutoffs, rerun=False):
    """
    Compute ion pair residence times.

    Parameters:
        seps (list): List of carbon slit heights.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        cip_cutoffs (numpy.ndarray): Cutoff distances for contact ion pairs.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/ion_pair_residence_times.npy"
    if rerun == True or not os.path.exists(filename):
        ion_pair_residence_times = np.empty((len(seps), len(runs)), dtype=object)
        ion_pair_acfs = np.empty((len(seps), len(runs)), dtype=object)
        times_all = np.empty((len(seps), len(runs)), dtype=object)
        for j, sep in enumerate(seps):
            for i, run in enumerate(runs):
                path = f"sep{sep}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                times = generate_times(u, run_start)
                times_all[j, i] = times
                (
                    ion_pair_residence_times[j, i],
                    ion_pair_acfs[j, i],
                ) = ion_pairing.calc_neigh_corr(
                    path,
                    u,
                    u.select_atoms("name Na"),
                    times,
                    cutoff_dist=cip_cutoffs[j],
                    run_start=run_start,
                    rerun=rerun,
                )
        ion_pair_residence_avg = np.empty((len(seps)), dtype=object)
        ion_pair_residence_err = np.empty((len(seps)), dtype=object)
        ion_pair_acf = np.empty((len(seps)), dtype=object)
        ion_pair_acf_err = np.empty((len(seps)), dtype=object)
        for j, sep in enumerate(seps):
            (
                ion_pair_residence_avg[j],
                ion_pair_residence_err[j],
            ) = ion_pairing.average_replicates(ion_pair_residence_times[j, :])
            min_len = np.min([len(ion_pair_acfs[j, i]) for i in range(len(runs))])
            ion_pair_acf[j] = np.mean(
                [ion_pair_acfs[j, i][:min_len] for i in range(len(runs))], axis=0
            )
            ion_pair_acf_err[j] = np.std(
                [ion_pair_acfs[j, i][:min_len] for i in range(len(runs))], axis=0
            )

        np.save("data/ion_pair_residence_times.npy", ion_pair_residence_avg)
        np.save("data/ion_pair_residence_times_err.npy", ion_pair_residence_err)
        np.save("data/ion_pair_acf.npy", ion_pair_acf)
        np.save("data/ion_pair_acf_err.npy", ion_pair_acf_err)
        np.save("data/times.npy", times_all)


def run_coordination_analysis(seps, runs, run_start, rerun=False):
    """
    Perform coordination number analysis.

    Parameters:
        seps (list): List of carbon slit heights.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/coordination_numbers.npy"
    if rerun == True or not os.path.exists(filename):
        coordination_numbers = np.empty((len(seps), len(runs), 2), dtype=object)
        coordination_numbers_err = np.empty((len(seps), len(runs), 2), dtype=object)
        coordination_bins = np.empty((len(seps)), dtype=object)
        for j, sep in enumerate(seps):
            for i, run in enumerate(runs):
                path = f"sep{sep}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                (
                    coordination_bins[j],
                    coordination_numbers[j, i, 0],
                    coordination_numbers[j, i, 1],
                    coordination_numbers_err[j, i, 0],
                    coordination_numbers_err[j, i, 1],
                ) = ion_pairing.coordination_analysis(
                    path, u, seps, run_start=run_start, rerun=rerun
                )
        np.save("data/coordination_numbers.npy", coordination_numbers)
        np.save("data/coordination_numbers_err.npy", coordination_numbers_err)
        np.save("data/coordination_bins.npy", coordination_bins)


def compute_2d_density(runs):
    """
    Compute 2D free energy profiles for each species on the graphene lattice in the H=6.7 Angstrom slit.

    Parameters:
        runs (list): List of run indices.
    """
    j = 0
    sep = 6.7
    path = f"sep{sep}/run0/"
    u = create_mda(
        path,
        data_file="../run_initial/end_equil.data",
        dcd_file="traj_unwrapped_*.dcd",
        cat_dcd=True,
    )

    x_dim = 2 * 2.47
    y_dim = 1 * 4.278

    n_x = int(round(u.dimensions[0] / x_dim))
    n_y = int(round(u.dimensions[1] / y_dim))

    carbon_positions_split = np.empty((n_x, n_y), dtype=object)

    n_bins = 20

    positions_cl_hist = np.zeros((n_bins, int(n_bins * y_dim / x_dim)), dtype=float)
    positions_na_hist = np.zeros((n_bins, int(n_bins * y_dim / x_dim)), dtype=float)
    positions_o_hist = np.zeros((n_bins, int(n_bins * y_dim / x_dim)), dtype=float)

    for i, run in enumerate(runs):
        path = f"sep{sep}/run{run}/"
        u = create_mda(
            path,
            data_file="../run_initial/end_equil.data",
            dcd_file="traj_unwrapped_*.dcd",
            cat_dcd=True,
        )
        cation_positions = (
            density_analysis.get_positions(path, u, u.select_atoms("name Na"), False)
            % u.dimensions[:3]
        )
        anion_positions = (
            density_analysis.get_positions(path, u, u.select_atoms("name Cl"), False)
            % u.dimensions[:3]
        )
        oxygen_positions = (
            density_analysis.get_positions(path, u, u.select_atoms("name O"), False)
            % u.dimensions[:3]
        )
        carbon_positions = density_analysis.get_positions(
            path, u, u.select_atoms("name C"), False
        )

        for i in range(n_x):
            for j in range(n_y):
                xmin = x_dim * i
                xmax = x_dim * (i + 1)
                ymin = y_dim * j
                ymax = y_dim * (j + 1)

                condition = np.logical_and(
                    np.logical_and(
                        carbon_positions[0, :, 0] > xmin,
                        carbon_positions[0, :, 0] < xmax,
                    ),
                    np.logical_and(
                        carbon_positions[0, :, 1] > ymin,
                        carbon_positions[0, :, 1] < ymax,
                    ),
                )
                carbon_positions_split[i, j] = carbon_positions[0, :, :][condition]

            condition = np.logical_and(
                np.logical_and(
                    anion_positions[:, :, 0] > xmin, anion_positions[:, :, 0] < xmax
                ),
                np.logical_and(
                    anion_positions[:, :, 1] > ymin, anion_positions[:, :, 1] < ymax
                ),
            )
            positions_cl_split = anion_positions[condition]

            condition = np.logical_and(
                np.logical_and(
                    cation_positions[:, :, 0] > xmin, cation_positions[:, :, 0] < xmax
                ),
                np.logical_and(
                    cation_positions[:, :, 1] > ymin, cation_positions[:, :, 1] < ymax
                ),
            )
            positions_na_split = cation_positions[condition]

            condition = np.logical_and(
                np.logical_and(
                    oxygen_positions[:, :, 0] > xmin, oxygen_positions[:, :, 0] < xmax
                ),
                np.logical_and(
                    oxygen_positions[:, :, 1] > ymin, oxygen_positions[:, :, 1] < ymax
                ),
            )
            positions_o_split = oxygen_positions[condition]

            positions_cl_hist_temp, xedges, yedges = np.histogram2d(
                positions_cl_split[:, 0],
                positions_cl_split[:, 1],
                bins=[n_bins, int(n_bins * y_dim / x_dim)],
            )
            positions_na_hist_temp, xedges, yedges = np.histogram2d(
                positions_na_split[:, 0],
                positions_na_split[:, 1],
                bins=[n_bins, int(n_bins * y_dim / x_dim)],
            )
            positions_o_hist_temp, xedges, yedges = np.histogram2d(
                positions_o_split[:, 0],
                positions_o_split[:, 1],
                bins=[n_bins, int(n_bins * y_dim / x_dim)],
            )

            positions_cl_hist += positions_cl_hist_temp
            positions_na_hist += positions_na_hist_temp
            positions_o_hist += positions_o_hist_temp

    H_all = [
        -np.log(positions_o_hist.T),
        -np.log(positions_cl_hist.T),
        -np.log(positions_na_hist.T),
    ]
    H_shifted = []
    for H in H_all:
        H_shifted.append(H - np.min(H))

    np.save("data/2d_density.npy", H_shifted)
    np.save("data/carbon_positions_split.npy", carbon_positions_split)
    np.save(
        "data/2d_density_extent.npy",
        np.array([xedges, yedges, xmin, ymin], dtype=object),
    )


def compute_adsorption(seps, runs, na_dist, cl_dist, run_start, rerun=False):
    """
    Compute fractions of each ion adsorbed to carbon.

    Parameters:
        seps (list): List of carbon slit heights.
        runs (list): List of replicate run indices.
        na_dist (numpy.ndarray): Cutoff distances for sodium ion adsorption (first minimum of density profile).
        cl_dist (numpy.ndarray): Cutoff distances for chloride adsorption (first minimum of density profile).
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/adsorption.npy"
    if rerun == True or not os.path.exists(filename):
        rerun = False
        adsorption = np.empty((2, len(seps), len(runs)), dtype=object)
        adsorption_err = np.empty((2, len(seps), len(runs)), dtype=object)
        for j, sep in enumerate(seps):
            for i, run in enumerate(runs):
                path = f"sep{sep}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                adsorption[0, j, i], _ = ion_pairing.compute_ionAdsorptionFrac(
                    path,
                    u.select_atoms("name Na"),
                    u,
                    dist=na_dist[j],
                    skip=100,
                    run_start=run_start,
                    rerun=rerun,
                )
                adsorption[1, j, i], _ = ion_pairing.compute_ionAdsorptionFrac(
                    path,
                    u.select_atoms("name Cl"),
                    u,
                    dist=cl_dist[j],
                    skip=100,
                    run_start=run_start,
                    rerun=rerun,
                )
        adsorption_avg = np.empty((2, len(seps)), dtype=object)
        adsorption_err = np.empty((2, len(seps)), dtype=object)
        for j, sep in enumerate(seps):
            adsorption_avg[0, j] = np.mean(adsorption[0, j, :])
            adsorption_avg[1, j] = np.mean(adsorption[1, j, :])
        adsorption_err[0, :] = [np.std(adsorption[0, j, :]) for j in range(len(seps))]
        adsorption_err[1, :] = [np.std(adsorption[1, j, :]) for j in range(len(seps))]
        np.save("data/adsorption.npy", adsorption_avg)
        np.save("data/adsorption_err.npy", adsorption_err)


def analyze_all():
    """
    Perform all analysis steps.
    """
    print("PMF")
    run_pmf_analysis(seps, runs, run_start, rerun=False)

    print("Density profiles")
    run_density_profiles(seps, runs, run_start, rerun=False)

    print("Ion pairing fractions")
    # load cutoff distances, separately saved by analyzing the PMFs computed above
    cip_dists = np.load("data/cip_cutoffs.npy")
    ssip_dists = np.load("data/ssip_cutoffs.npy")
    run_ion_pairing_fracs(seps, runs, cip_dists, ssip_dists, run_start, rerun=False)

    print("Ion pair residence times")
    run_ion_pair_residence_times(seps, runs, run_start, cip_dists, rerun=False)

    print("Coordination numbers")
    run_coordination_analysis(seps, runs, run_start, rerun=False)

    print("2D Free energy profiles")
    compute_2d_density(runs)

    print("Adsorption fractions")
    # load cutoff distances, separately saved by analyzing the density profiles computed above
    na_dist = np.load("data/na_c_dist.npy")
    cl_dist = np.load("data/cl_c_dist.npy")
    compute_adsorption(seps, runs, na_dist, cl_dist, run_start, rerun=True)


analyze_all()
