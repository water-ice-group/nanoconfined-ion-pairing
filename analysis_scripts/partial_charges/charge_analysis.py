import numpy as np
import MDAnalysis as mda
import pandas as pd
import os
import scipy
import scipy.stats

# prior to running this script, the Bader charges for each frame must be saved in a .txt or .npy file


def get_charges_and_positions(systems, name, runs, n_calcs_per_run=100, n_carbons=120):
    """Get charges and positions for a single system.

    Parameters
    ----------
    systems : list of str
            List of systems to analyze.
    name : str
            Name of the charge method used (e.g. 'bader').
    runs : array of int
            Array of runs to analyze.
    n_calcs_per_run : int, optional
            Number of structures analyzed per run. The default is 100.
    n_carbons : int, optional
            Number of carbons in the system. The default is 120.

    Returns
    -------
    charges : np.ndarray
            Array of charges.
    positions : np.ndarray
            Array of positions.

    """

    n_calcs = n_calcs_per_run * len(runs)
    c_charges = np.empty((len(systems), n_calcs, n_carbons), dtype=float)
    na_charges = np.empty((len(systems), n_calcs), dtype=float)
    cl_charges = np.empty((len(systems), n_calcs), dtype=float)

    c_positions = np.empty((len(systems), n_calcs, n_carbons, 3), dtype=float)
    na_positions = np.empty((len(systems), n_calcs, 3), dtype=float)
    cl_positions = np.empty((len(systems), n_calcs, 3), dtype=float)

    for j in range(len(systems)):
        system = systems[j]
        for k, run in enumerate(runs):
            for i in range(n_calcs_per_run):
                index = i + n_calcs_per_run * k
                path = system + f"run{run}_frame{i}"
                if os.path.exists(path + f"/{name}_charges.txt"):
                    charges = np.loadtxt(path + f"/{name}_charges.txt")
                else:
                    charges = np.load(path + f"/{name}-charges.npy")
                c_charges[j, index, :] = charges[:n_carbons]
                na_charges[j, index] = charges[-2]
                cl_charges[j, index] = charges[-1]

                with open(path + "/trajectory-input.xyz") as f:
                    lines = f.readlines()
                    na_positions[j, index, :] = lines[-2].split()[1:]
                    cl_positions[j, index, :] = lines[-1].split()[1:]
                    for l in range(n_carbons):
                        c_positions[j, index, l, :] = lines[l + 2].split()[1:]

    charges = [c_charges, na_charges, cl_charges]
    positions = [c_positions, na_positions, cl_positions]

    np.save(f"data/{name}_c_charges.npy", c_charges)

    return charges, positions


def ion_charge_vs_z(seps, positions, name, charges, nbins=50):
    """
    Compute ion charges versus distance along the z-axis.

    Parameters:
    - seps (list): List of carbon slit heights.
    - positions (array): Array containing positions of ions.
    - name (str): Name of the charge method used (e.g. 'bader').
    - charges (array): Array containing charges of ions.
    - nbins (int, optional): Number of bins for binning charge distributions (default is 50).

     Returns:
    - None
    """
    na_positions = positions[1]
    cl_positions = positions[2]
    na_charges = charges[1]
    cl_charges = charges[2]

    ion_charge_bins = np.empty(len(seps), dtype=object)
    ion_charge_mean = np.empty(len(seps), dtype=object)
    ion_charge_err = np.empty(len(seps), dtype=object)

    for j in range(len(seps)):
        midpoint = 7.5 + seps[j] / 2

        mean_charge_na, bin_edges_na, binnumber = scipy.stats.binned_statistic(
            np.abs(na_positions[j, :, 2] - midpoint).flatten(),
            na_charges[j, :].flatten(),
            bins=nbins,
        )
        err_charge_na, bin_edges_na, binnumber = scipy.stats.binned_statistic(
            np.abs(na_positions[j, :, 2] - midpoint).flatten(),
            na_charges[j, :].flatten(),
            statistic="std",
            bins=nbins,
        )

        mean_charge_cl, bin_edges_cl, binnumber = scipy.stats.binned_statistic(
            np.abs(cl_positions[j, :, 2] - midpoint).flatten(),
            cl_charges[j, :].flatten(),
            bins=nbins,
        )
        err_charge_cl, bin_edges_cl, binnumber = scipy.stats.binned_statistic(
            np.abs(cl_positions[j, :, 2] - midpoint).flatten(),
            cl_charges[j, :].flatten(),
            statistic="std",
            bins=nbins,
        )

        # remove nan from empty bins
        mean_charge_na_tmp = mean_charge_na
        mean_charge_cl_tmp = mean_charge_cl
        mean_charge_na = mean_charge_na[
            np.argwhere(~np.isnan(mean_charge_na_tmp))[:, 0]
        ]
        err_charge_na = err_charge_na[np.argwhere(~np.isnan(mean_charge_na_tmp))[:, 0]]

        mean_charge_cl = mean_charge_cl[
            np.argwhere(~np.isnan(mean_charge_cl_tmp))[:, 0]
        ]
        err_charge_cl = err_charge_cl[np.argwhere(~np.isnan(mean_charge_cl_tmp))[:, 0]]

        bin_centers_na = []
        for i in range(len(bin_edges_na) - 1):
            bin_centers_na.append((bin_edges_na[i + 1] + bin_edges_na[i]) / 2)
        bin_centers_na = np.array(bin_centers_na)

        bin_centers_cl = []
        for i in range(len(bin_edges_cl) - 1):
            bin_centers_cl.append((bin_edges_cl[i + 1] + bin_edges_cl[i]) / 2)
        bin_centers_cl = np.array(bin_centers_cl)

        bin_centers_na = bin_centers_na[
            np.argwhere(~np.isnan(mean_charge_na_tmp))[:, 0]
        ]
        bin_centers_cl = bin_centers_cl[
            np.argwhere(~np.isnan(mean_charge_cl_tmp))[:, 0]
        ]

        ion_charge_bins[j] = [bin_centers_na, bin_centers_cl]
        ion_charge_mean[j] = [mean_charge_na, mean_charge_cl]
        ion_charge_err[j] = [err_charge_na, err_charge_cl]

    np.save(f"data/{name}_ion_charge_bins.npy", ion_charge_bins)
    np.save(f"data/{name}_ion_charge_mean.npy", ion_charge_mean)
    np.save(f"data/{name}_ion_charge_err.npy", ion_charge_err)


def carbon_charges_near_ions(
    systems,
    positions,
    charges,
    name,
    na_c_dist,
    cl_c_dist,
    n_calcs=1000,
    n_carbons=120,
    Lx=12.35,
    Ly=12.834,
    nbins=150,
):
    """
    Compute the distribution of carbon charges near adsorbed ions.

    Parameters:
    - systems (list): List of system names.
    - positions (array): Array containing positions of ions and carbon atoms.
    - charges (array): Array containing charges of ions and carbon atoms.
    - name (str): Name of the charge method used (e.g. 'bader').
    - na_c_dist (array): Cutoff distances for sodium ion adsorption (first minimum of density profile).
    - cl_c_dist (array): Cutoff distances for chloride adsorption (first minimum of density profile).
    - n_calcs (int, optional): Number of calculations (default is 1000).
    - n_carbons (int, optional): Number of carbon atoms in each system (default is 120).
    - Lx (float, optional): Length of the simulation box along x-axis in Angstroms (default is 12.35).
    - Ly (float, optional): Length of the simulation box along y-axis in Angstroms (default is 12.834).
    - nbins (int, optional): Number of bins for binning charge distributions (default is 150).

    Returns:
    - None
    """

    c_charge_bins = np.empty(len(systems), dtype=object)
    c_charge_mean = np.empty(len(systems), dtype=object)
    c_charge_err = np.empty(len(systems), dtype=object)

    na_positions = positions[1]
    cl_positions = positions[2]
    c_positions = positions[0]

    na_charges = charges[1]
    cl_charges = charges[2]
    c_charges = charges[0]

    top_c_pos = np.empty(len(systems))
    bot_c_pos = np.empty(len(systems))
    for j in range(len(systems)):
        top_c_pos[j] = c_positions[j, 0, 4, 2]
        bot_c_pos[j] = c_positions[j, 0, 0, 2]

    adsorbed_cations = np.empty(len(systems), dtype=object)
    adsorbed_anions = np.empty(len(systems), dtype=object)

    adsorbed_cat_indices = np.empty(len(systems), dtype=object)
    adsorbed_an_indices = np.empty(len(systems), dtype=object)

    adsorbed_cat_charges = np.empty(len(systems), dtype=object)
    adsorbed_an_charges = np.empty(len(systems), dtype=object)

    free_cat_charges = np.empty(len(systems), dtype=object)
    free_an_charges = np.empty(len(systems), dtype=object)

    carb_charges_nearNa = np.empty(len(systems), dtype=object)
    carb_charges_nearCl = np.empty(len(systems), dtype=object)

    carb_Na_distances = np.empty(len(systems), dtype=object)
    carb_Cl_distances = np.empty(len(systems), dtype=object)

    for j in range(len(systems)):
        # get ions adsorbed to carbon
        if na_c_dist[j] == 100.0:  # all ions are adsorbed
            adsorbed_cations[j] = na_positions[j]
            adsorbed_cat_indices[j] = np.arange(0, n_calcs)
            free_cat_charges[j] = []
        else:
            adsorbed_cat_indices1 = np.where(
                na_positions[j][:, 2] < bot_c_pos[j] + na_c_dist[j]
            )[0]
            adsorbed_cat_indices2 = np.where(
                na_positions[j][:, 2] > top_c_pos[j] - na_c_dist[j]
            )[0]
            adsorbed_cat_indices[j] = np.concatenate(
                (adsorbed_cat_indices1, adsorbed_cat_indices2)
            )

            adsorbed_cations[j] = na_positions[j][adsorbed_cat_indices[j]]
            free_cat_indices = np.setdiff1d(
                np.arange(len(na_positions[j])), adsorbed_cat_indices[j]
            )
            free_cat_charges[j] = na_charges[j][free_cat_indices]

        adsorbed_cat_charges[j] = na_charges[j][adsorbed_cat_indices[j]]

        if cl_c_dist[j] == 100.0:  # all ions are adsorbed
            adsorbed_anions[j] = cl_positions[j]
            adsorbed_an_indices[j] = np.arange(0, n_calcs)
            free_an_charges[j] = []
        else:
            adsorbed_an_indices1 = np.where(
                cl_positions[j][:, 2] < bot_c_pos[j] + cl_c_dist[j]
            )[0]
            adsorbed_an_indices2 = np.where(
                cl_positions[j][:, 2] > top_c_pos[j] - cl_c_dist[j]
            )[0]
            adsorbed_an_indices[j] = np.concatenate(
                (adsorbed_an_indices1, adsorbed_an_indices2)
            )
            free_an_indices = np.setdiff1d(
                np.arange(len(cl_positions[j])), adsorbed_an_indices[j]
            )
            free_an_charges[j] = cl_charges[j][free_an_indices]

        adsorbed_an_charges[j] = cl_charges[j][adsorbed_an_indices[j]]

        # for each adorbed ion, get distribution of nearby charges
        carb_charges_nearNa[j] = np.empty(
            (len(adsorbed_cat_indices[j]), n_carbons), dtype=float
        )
        carb_Na_distances[j] = np.empty(
            (len(adsorbed_cat_indices[j]), n_carbons), dtype=float
        )

        carb_charges_nearCl[j] = np.empty(
            (len(adsorbed_an_indices[j]), n_carbons), dtype=float
        )
        carb_Cl_distances[j] = np.empty(
            (len(adsorbed_an_indices[j]), n_carbons), dtype=float
        )

        for idx, i in enumerate(adsorbed_cat_indices[j]):
            na_x = na_positions[j, i, 0] % Lx
            na_y = na_positions[j, i, 1] % Ly
            na_z = na_positions[j, i, 2]

            c_charges_temp = []
            c_indices = []
            for k in range(n_carbons):
                c_x = c_positions[j, i, k, 0]
                c_y = c_positions[j, i, k, 1]
                c_z = c_positions[j, i, k, 2]

                distx = na_x - c_x
                disty = na_y - c_y

                if distx > 0.5 * Lx:
                    distx -= Lx
                elif distx < -0.5 * Lx:
                    distx += Lx
                if disty > 0.5 * Ly:
                    disty -= Ly
                elif disty < -0.5 * Ly:
                    disty += Ly

                dist = np.sqrt(distx**2 + disty**2)
                # print(i,j,k)
                carb_Na_distances[j][idx, k] = dist

        for idx, i in enumerate(adsorbed_an_indices[j]):
            cl_x = cl_positions[j, i, 0] % Lx
            cl_y = cl_positions[j, i, 1] % Ly
            cl_z = cl_positions[j, i, 2]

            c_charges_temp = []
            c_indices = []
            for k in range(n_carbons):
                c_x = c_positions[j, i, k, 0]
                c_y = c_positions[j, i, k, 1]
                c_z = c_positions[j, i, k, 2]

                distx = cl_x - c_x
                disty = cl_y - c_y

                if distx > 0.5 * Lx:
                    distx -= Lx
                elif distx < -0.5 * Lx:
                    distx += Lx
                if disty > 0.5 * Ly:
                    disty -= Ly
                elif disty < -0.5 * Ly:
                    disty += Ly

                dist = np.sqrt(distx**2 + disty**2)
                carb_Cl_distances[j][idx, k] = dist

    for j in range(len(systems)):
        mean_charge_na, bin_edges_na, binnumber = scipy.stats.binned_statistic(
            carb_Na_distances[j].flatten(),
            c_charges[j, adsorbed_cat_indices[j], :].flatten(),
            bins=nbins,
        )
        err_charge_na, bin_edges_na, binnumber = scipy.stats.binned_statistic(
            carb_Na_distances[j].flatten(),
            c_charges[j, adsorbed_cat_indices[j], :].flatten(),
            statistic="std",
            bins=nbins,
        )

        mean_charge_cl, bin_edges_cl, binnumber = scipy.stats.binned_statistic(
            carb_Cl_distances[j].flatten(),
            c_charges[j, adsorbed_an_indices[j], :].flatten(),
            bins=nbins,
        )
        err_charge_cl, bin_edges_cl, binnumber = scipy.stats.binned_statistic(
            carb_Cl_distances[j].flatten(),
            c_charges[j, adsorbed_an_indices[j], :].flatten(),
            statistic="std",
            bins=nbins,
        )

        # remove nan from empty bins
        mean_charge_na_tmp = mean_charge_na
        mean_charge_cl_tmp = mean_charge_cl
        mean_charge_na = mean_charge_na[
            np.argwhere(~np.isnan(mean_charge_na_tmp))[:, 0]
        ]
        err_charge_na = err_charge_na[np.argwhere(~np.isnan(mean_charge_na_tmp))[:, 0]]

        mean_charge_cl = mean_charge_cl[
            np.argwhere(~np.isnan(mean_charge_cl_tmp))[:, 0]
        ]
        err_charge_cl = err_charge_cl[np.argwhere(~np.isnan(mean_charge_cl_tmp))[:, 0]]

        bin_centers_na = []
        for i in range(len(bin_edges_na) - 1):
            bin_centers_na.append((bin_edges_na[i + 1] + bin_edges_na[i]) / 2)
        bin_centers_na = np.array(bin_centers_na)

        bin_centers_cl = []
        for i in range(len(bin_edges_cl) - 1):
            bin_centers_cl.append((bin_edges_cl[i + 1] + bin_edges_cl[i]) / 2)
        bin_centers_cl = np.array(bin_centers_cl)

        bin_centers_na = bin_centers_na[
            np.argwhere(~np.isnan(mean_charge_na_tmp))[:, 0]
        ]
        bin_centers_cl = bin_centers_cl[
            np.argwhere(~np.isnan(mean_charge_cl_tmp))[:, 0]
        ]

        c_charge_bins[j] = [bin_centers_na, bin_centers_cl]
        c_charge_mean[j] = [mean_charge_na, mean_charge_cl]
        c_charge_err[j] = [err_charge_na, err_charge_cl]

    np.save(f"data/{name}_c_charge_bins.npy", c_charge_bins)
    np.save(f"data/{name}_c_charge_mean.npy", c_charge_mean)
    np.save(f"data/{name}_c_charge_err.npy", c_charge_err)


systems = ["sep6.7/", "sep10.05/", "sep13.4/", "sep16.75/"]
seps = [6.7, 10.05, 13.4, 16.75]
runs = np.arange(0, 10)

# load distance cutoffs for ions to be considered adsorbed to the carbon
na_c_dist = np.load("../../data/nnp_md/confined/na_c_dist.npy", allow_pickle=True)
cl_c_dist = np.load("../../data/nnp_md/confined/cl_c_dist.npy", allow_pickle=True)

# run Bader charge analysis
bader_charges, positions = get_charges_and_positions(systems, "bader", runs)
ion_charge_vs_z(seps, positions, "bader", bader_charges)
carbon_charges_near_ions(
    systems, positions, bader_charges, "bader", na_c_dist, cl_c_dist
)
