import numpy as np
import MDAnalysis as mda
import os
import re
from scipy import interpolate
from scipy.spatial.distance import cdist
from scipy import stats


def compute_ionPairFrac(
    path, run, cip_dist=3.85, ssip_dist=6.25, skip=1, run_start=0, rerun=False
):
    """
    Compute the fraction of ions that are in CIPs (Contact Ion Pairs), SSIPs (Solvent-Separated Ion Pairs),
    and free ions based on given cutoff distances.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        cip_dist (float, optional): Cutoff distance for CIPs. Default is 3.85 Angstrom.
        ssip_dist (float, optional): Cutoff distance for SSIPs. Default is 6.25 Angstrom.
        skip (int, optional): Number of frames to skip in trajectory analysis. Default is 1.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing average fraction of ions in CIPs, SSIPs, and free ions,
               and an array of fractions for each frame.
    """
    filename = path + f"ion_pairing.npy"
    if rerun == True or not os.path.exists(filename):
        na_ions = run.select_atoms("name Na")

        frac_cip = []
        frac_free = []
        frac_ssip = []

        for ts in run.trajectory[run_start:-1:skip]:
            free_na = run.select_atoms(f"name Na and not around {ssip_dist} name Cl")
            bound_na = run.select_atoms(f"name Na and around {cip_dist} name Cl")

            frac_cip.append(bound_na.atoms.n_atoms / float(na_ions.atoms.n_atoms))
            frac_free.append(free_na.atoms.n_atoms / float(na_ions.atoms.n_atoms))
            frac_ssip.append(1 - frac_cip[-1] - frac_free[-1])

        avg_frac_cip = np.mean(np.asarray(frac_cip))
        avg_frac_ssip = np.mean(np.asarray(frac_ssip))
        avg_frac_free = 1 - avg_frac_cip - avg_frac_ssip

        avg_pairing = [avg_frac_cip, avg_frac_ssip, avg_frac_free]
        pairing = [frac_cip, frac_ssip, frac_free]

        np.save(path + "ion_pairing.npy", pairing)
        np.save(path + "avg_ion_pairing.npy", avg_pairing)

    else:
        pairing = np.load(path + "ion_pairing.npy")
        avg_pairing = np.load(path + "avg_ion_pairing.npy")

    return avg_pairing, pairing


def get_ion_pair_stats(
    path, run, run_start=0, skip=1, nao_dist=3.36045, clh_dist=2.9337, rerun=False
):
    """
    Compute various statistics related to ion pairs such as distances, heights, orientations, and coordinations.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        skip (int, optional): Number of frames to skip in trajectory analysis. Default is 1.
        nao_dist (float, optional): Distance cutoff for Na-O coordination. Default is 3.36045 Angstrom.
        clh_dist (float, optional): Distance cutoff for Cl-H coordination. Default is 2.9337 Angstrom.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing arrays of ion pair distances, heights, orientations, Na heights, Cl heights,
               Na-O coordinations, and Cl-H coordinations.
    """

    filename = path + f"ion_pair_distances.npy"
    if rerun == True or not os.path.exists(filename):
        na_atoms = run.select_atoms("name Na")
        cl_atoms = run.select_atoms("name Cl")
        ion_pair_distances = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )
        ion_pair_heights = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )
        na_heights = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )
        cl_heights = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )
        ion_pair_orientations = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )
        nao_coordination = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )
        clh_coordination = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(na_atoms), len(cl_atoms)),
            dtype=float,
        )

        bot_c_pos = np.min(run.atoms.positions[:, 2])
        top_c_pos = np.max(run.atoms.positions[:, 2])
        for i, ts in enumerate(run.trajectory[run_start:-1:skip]):
            # I checked that this function gave the same results as below
            # ion_pair_distances[i,:] = MDAnalysis.analysis.distances.distance_array(na_atoms.positions, cl_atoms.positions, box=run.dimensions).flatten()

            # get vector separating each pair of ions using minimum image convention
            dists = np.empty((3, len(na_atoms), len(cl_atoms)))
            for j in range(3):
                dists[j, :, :] = cdist(
                    (na_atoms.positions[:, j] % run.dimensions[j]).reshape(-1, 1),
                    (cl_atoms.positions[:, j] % run.dimensions[j]).reshape(-1, 1),
                )
                dists[j, :, :] = np.where(
                    dists[j, :, :] > (run.dimensions[j] / 2)[..., None],
                    dists[j, :, :] - run.dimensions[j][..., None],
                    dists[j, :, :],
                )

            for j in range(len(na_atoms)):
                nao_coord = run.select_atoms(
                    f"name O and around {nao_dist} index {na_atoms[j].index}"
                ).n_atoms
                nao_coordination[i, j, :] = np.repeat(nao_coord, len(cl_atoms))
                for k in range(len(cl_atoms)):
                    if j == 0:
                        clh_coord = run.select_atoms(
                            f"name H and around {clh_dist} index {cl_atoms[k].index}"
                        ).n_atoms
                        clh_coordination[i, :, k] = np.repeat(clh_coord, len(na_atoms))
                    ion_pair_distances[i, j, k] = np.linalg.norm(dists[:, j, k])
                    ion_pair_orientations[i, j, k] = np.dot(
                        [0, 0, 1], dists[:, j, k]
                    ) / np.linalg.norm(dists[:, j, k])
                    avg_z_pos = 0.5 * (
                        na_atoms.positions[j, 2] + cl_atoms.positions[j, 2]
                    )
                    ion_pair_heights[i, j, k] = np.min(
                        [avg_z_pos - bot_c_pos, top_c_pos - avg_z_pos]
                    )
                    na_heights[i, j, k] = np.min(
                        [
                            na_atoms.positions[j, 2] - bot_c_pos,
                            top_c_pos - na_atoms.positions[j, 2],
                        ]
                    )
                    cl_heights[i, j, k] = np.min(
                        [
                            cl_atoms.positions[k, 2] - bot_c_pos,
                            top_c_pos - cl_atoms.positions[k, 2],
                        ]
                    )
        np.save(path + "ion_pair_distances.npy", ion_pair_distances)
        np.save(path + "ion_pair_heights.npy", ion_pair_heights)
        np.save(path + "ion_pair_orientations.npy", ion_pair_orientations)
        np.save(path + "na_heights.npy", na_heights)
        np.save(path + "cl_heights.npy", cl_heights)
        np.save(path + "nao_coordination.npy", nao_coordination)
        np.save(path + "clh_coordination.npy", clh_coordination)

    else:
        ion_pair_distances = np.load(path + "ion_pair_distances.npy")
        ion_pair_heights = np.load(path + "ion_pair_heights.npy")
        ion_pair_orientations = np.load(path + "ion_pair_orientations.npy")
        na_heights = np.load(path + "na_heights.npy")
        cl_heights = np.load(path + "cl_heights.npy")
        nao_coordination = np.load(path + "nao_coordination.npy")
        clh_coordination = np.load(path + "clh_coordination.npy")

    return (
        ion_pair_distances,
        ion_pair_heights,
        ion_pair_orientations,
        na_heights,
        cl_heights,
        nao_coordination,
        clh_coordination,
    )


def coordination_analysis(path, run, run_start=0, rerun=False):
    """
    Compute the coordination numbers of Na-O and Cl-H pairs.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing arrays of bin edges, mean Na-O coordination, mean Cl-H coordination,
               Na-O coordination standard deviation, and Cl-H coordination standard deviation.
    """
    filename = path + f"nao_vs_r.npy"
    if rerun == True or not os.path.exists(filename):
        (
            ion_pair_distances,
            ion_pair_heights,
            ion_pair_orientations,
            na_heights,
            cl_heights,
            nao_coordination,
            clh_coordination,
        ) = get_ion_pair_stats(path, run, run_start=run_start, rerun=rerun)

        nbins = 300

        nao_vs_r, bins, binnumber = stats.binned_statistic(
            np.concatenate(ion_pair_distances).flatten(),
            np.concatenate(nao_coordination).flatten(),
            statistic="mean",
            bins=nbins,
        )
        nao_err, _, binnumber = stats.binned_statistic(
            np.concatenate(ion_pair_distances).flatten(),
            np.concatenate(nao_coordination).flatten(),
            statistic="std",
            bins=nbins,
        )

        clh_vs_r, _, binnumber = stats.binned_statistic(
            np.concatenate(ion_pair_distances).flatten(),
            np.concatenate(clh_coordination).flatten(),
            statistic="mean",
            bins=nbins,
        )
        clh_err, _, binnumber = stats.binned_statistic(
            np.concatenate(ion_pair_distances).flatten(),
            np.concatenate(clh_coordination).flatten(),
            statistic="std",
            bins=nbins,
        )

        np.save(path + "nao_vs_r.npy", nao_vs_r)
        np.save(path + "nao_err.npy", nao_err)
        np.save(path + "bins.npy", bins)
        np.save(path + "clh_vs_r.npy", clh_vs_r)
        np.save(path + "clh_err.npy", clh_err)

    else:
        nao_vs_r = np.load(path + "nao_vs_r.npy")
        nao_err = np.load(path + "nao_err.npy")
        bins = np.load(path + "bins.npy")
        clh_vs_r = np.load(path + "clh_vs_r.npy")
        clh_err = np.load(path + "clh_err.npy")

    return bins, nao_vs_r, clh_vs_r, nao_err, clh_err


def get_pmf(path, run, sep, run_start, c_vdw=1.7, rerun=False):
    """
    Compute the potential of mean force (PMF) between ions in the system.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        sep (float): Separation distance between the two graphene sheets.
        run_start (int): Index of the starting frame for analysis.
        c_vdw (float, optional): carbon Van der Waals radius for calculating volume normalization. Default is 1.7 Angstrom.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing the PMF values and corresponding bin edges.
    """

    def get_area(path, run, rerun=False):
        """
        Calculate the area of the graphene sheet (the area of the x-y plane of the simulation box)

        Parameters:
            path (str): Path to the directory where output files will be saved.
            run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
            rerun (bool, optional): If True, recompute the area even if the output file exists. Default is False.

        Returns:
            float: Area of the graphene sheet.
        """
        filename = path + f"area.npy"
        if rerun == True or not os.path.exists(filename):
            area = run.dimensions[0] * run.dimensions[1]
            np.save(path + "area.npy", area)
        else:
            area = np.load(path + "area.npy")
        return area

    def get_volume_normalization(sep, ion_pair_distances, na_heights, c_vdw=1.7):
        """
        Calculate volume normalization factors for potential of mean force (PMF) calculation (A_shell).

        Parameters:
            sep (float): Separation distance between the two graphene sheets.
            ion_pair_distances (numpy.ndarray): Array of ion pair distances.
            na_heights (numpy.ndarray): Array of Na heights.

        Returns:
            numpy.ndarray: Volume normalization factors.
        """
        r = ion_pair_distances.flatten()
        h_top = na_heights.flatten() - c_vdw
        h_bot = sep - 2 * c_vdw - h_top

        cos_theta_top = np.where(h_top < r, h_top / r, 1)
        cos_theta_bot = np.where(h_bot < r, h_bot / r, 1)

        hist_vol_na = (
            4 * np.pi * r**2
            - 2 * np.pi * r**2 * (1 - cos_theta_top)
            - 2 * np.pi * r**2 * (1 - cos_theta_bot)
        )

        return hist_vol_na

    def compute_pmf(path, sep, ion_pair_distances, na_heights, area, c_vdw=1.7, dr=0.08):
        """
        Compute the potential of mean force (PMF) between ions in the system.

        Parameters:
            sep (float): Separation distance between the two graphene sheets.
            ion_pair_distances (numpy.ndarray): Array of ion pair distances.
            na_heights (numpy.ndarray): Array of Na heights.
            area (float): Area of the graphene sheet.
            c_vdw (float, optional): carbon Van der Waals radius for calculating volume normalization. Default is 1.7 Angstrom.
            dr (float, optional): Bin width for PMF calculation. Default is 0.08 Angstrom.

        Returns:
            tuple: A tuple containing the PMF values and corresponding bin edges.
        """
        bins = np.arange(1, np.sqrt(area), dr)
        weights = get_volume_normalization(sep, ion_pair_distances, na_heights, c_vdw)
        dens, edges = np.histogram(
            ion_pair_distances.flatten(), weights=1 / weights, bins=bins, density=False
        )
        edges = edges[:-1]
        bulk_dens = len(ion_pair_distances.flatten()) / (area * (sep - 2 * c_vdw))
        rdf = dens / dr / bulk_dens
        pmf_bins = edges
        pmf = -np.log(rdf)
        return pmf, pmf_bins

    filename = path + f"pmf.npy"
    if rerun == True or not os.path.exists(filename):
        area = get_area(path, run, rerun)
        (
            ion_pair_distances,
            ion_pair_heights,
            ion_pair_orientations,
            na_heights,
            cl_heights,
            nao_coordination,
            clh_coordination,
        ) = get_ion_pair_stats(path, run, run_start=run_start, rerun=rerun)
        pmf, pmf_bins = compute_pmf(
            path, sep, ion_pair_distances, na_heights, area, c_vdw, dr=0.08
        )
        np.save(path + "pmf.npy", pmf)
        np.save(path + "pmf_bins.npy", pmf_bins)
    else:
        pmf = np.load(path + "pmf.npy")
        pmf_bins = np.load(path + "pmf_bins.npy")
    return pmf, pmf_bins


def compute_ionAdsorptionFrac(path, atoms, run, dist, skip=1, run_start=0, rerun=False):
    """
    Compute the fraction of ions adsorbed onto the graphene surface.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        atoms (MDAnalysis.AtomGroup): AtomGroup representing the ions.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        dist (float): Cutoff distance for determining adsorption.
        skip (int, optional): Number of frames to skip in trajectory analysis. Default is 1.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing average fraction of ions adsorbed and an array of fractions for each frame.
    """
    atom_name = atoms[0].name
    filename = path + f"{atom_name}_adsorption.npy"
    if rerun == True or not os.path.exists(filename):
        frac_adsorbed = []

        for ts in run.trajectory[run_start:-1:skip]:
            bound_atoms = run.select_atoms(f"name {atom_name} and around {dist} name C")
            frac_adsorbed.append(bound_atoms.atoms.n_atoms / float(atoms.atoms.n_atoms))

        avg_frac_adsorbed = np.mean(np.asarray(frac_adsorbed))

        np.save(path + f"{atom_name}_adsorption.npy", frac_adsorbed)
        np.save(path + f"avg_{atom_name}_adsorption.npy", avg_frac_adsorbed)

    else:
        frac_adsorbed = np.load(path + f"{atom_name}_adsorption.npy")
        avg_frac_adsorbed = np.load(path + f"avg_{atom_name}_adsorption.npy")

    return avg_frac_adsorbed, frac_adsorbed


def calc_neigh_corr(path, run, atoms, times, run_start=0, cutoff_dist=3.6, rerun=False):
    """
    Calculate the ion pairing residence time and autocorrelation function.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        atoms (MDAnalysis.AtomGroup): AtomGroup representing the ions.
        times (np.ndarray): Array of time points corresponding to the trajectory.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        cutoff_dist (float, optional): Cutoff distance for defining ion pairs. Default is 3.6 Angstrom.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing the ion pairing decay time and normalized autocorrelation function.
    """

    def autocorrFFT(x):
        """
        Calculate the autocorrelation function using the fast Fourier transform.

        Parameters:
            x (numpy.ndarray): Array containing data.

        Returns:
            numpy.ndarray: Autocorrelation function.
        """
        N = len(x)
        F = np.fft.fft(x, n=2 * N)
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res = (res[:N]).real
        n = N * np.ones(N) - np.arange(0, N)
        return res / n

    def calc_acf(A_values):
        """
        Calculate the autocorrelation function for a set of adjacency matrices.

        Parameters:
            A_values (dict): Dictionary containing adjacency matrices.

        Returns:
            list: List of autocorrelation functions.
        """
        acfs = []
        for atomid, neighbors in A_values.items():
            atomid = int(re.search(r"\d+", atomid).group())
            acfs.append(autocorrFFT(neighbors))
        return acfs

    def get_decay_time(times, data, decay_value=1.0 / np.e):
        """
        Find the time required for the autocorrelation function to decay to a specified value.

        Parameters:
            times (numpy.ndarray): Array of time points.
            data (numpy.ndarray): Array of autocorrelation function values.
            decay_value (float, optional): Value at which the autocorrelation function is considered decayed.
                                        Default is 1.0/np.e.

        Returns:
            float: Time at which the autocorrelation function decays to the specified value.
        """
        f = interpolate.interp1d(times[: len(data)], data[: len(times)])
        times_new = np.linspace(times[0], times[-1], 2000)

        def loss(time):
            return abs(f(time) - decay_value)

        return min(times_new, key=loss)

    def neighbors(run, run_start, atom, cutoff_dist):
        """
        Find neighboring atoms within a specified distance for each atom in the system.

        Parameters:
            run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
            run_start (int): Index of the starting frame for analysis.
            atom (MDAnalysis.Atom): Atom for which neighbors are to be found.
            cutoff_dist (float): Cutoff distance for defining neighbors.

        Returns:
            dict: Dictionary containing adjacency matrices for neighboring atoms.
        """
        A_values = {}
        time_count = 0
        for ts in run.trajectory[run_start::]:
            shell = run.select_atoms(
                "(name Cl and around "
                + str(cutoff_dist)
                + " resid "
                + str(atom.resid)
                + ")"
            )
            # for each atom in shell, create/add to dictionary (key = atom id, value = list of values for step function)
            for shell_atom in shell.atoms:
                if str(shell_atom.id) not in A_values:
                    A_values[str(shell_atom.id)] = np.zeros(
                        int((run.trajectory.n_frames - run_start) / 1)
                    )
                A_values[str(shell_atom.id)][time_count] = 1
            time_count += 1

        # account for species exiting then re-entering the shell
        lag_period = 4 # if an atom leaves for less then this many frames and re-enters, it is not counted as leaving
        for key in A_values:
            started = False
            ended = False
            for i in range(len(A_values[key])):
                if A_values[key][i] == 1:
                    started = True
                if started and np.sum(A_values[key][i:i+lag_period]) == 0:
                    ended = True
                if ended:
                    A_values[key][i] = 0

        return A_values

    filename = path + f"ion_pairing_acf.npy"
    if rerun == True or not os.path.exists(filename):
        # Average ACFs for all cations
        acf_all = []
        for atom in atoms[:]:
            adjacency_matrix = neighbors(run, run_start, atom, cutoff_dist)
            acfs = calc_acf(adjacency_matrix)
            [acf_all.append(acf) for acf in acfs]
        acf_avg = np.mean(acf_all, axis=0)
        acf_avg_norm = acf_avg / acf_avg[0]

        decay_time = get_decay_time(times, acf_avg_norm)

        np.save(path + f"ion_pairing_decay_time.npy", decay_time)
        np.save(path + f"ion_pairing_acf.npy", acf_avg_norm)

    else:
        decay_time = np.load(path + f"ion_pairing_decay_time.npy")
        acf_avg_norm = np.load(path + f"ion_pairing_acf.npy")

    return decay_time, acf_avg_norm


def average_replicates(data, run_axis=0):
    """
    Compute the average and standard deviation of replicated data.

    Parameters:
        data (numpy.ndarray): Array of data with replicates along the specified axis
        run_axis (int, optional): The axis along which to compute the mean and standard deviation.
                                  Default is 0.

    Returns:
        tuple: A tuple containing the average and standard deviation of the data.
    """

    data_avg = np.mean(data, axis=run_axis)
    data_err = np.std(data, axis=run_axis)

    return data_avg, data_err
