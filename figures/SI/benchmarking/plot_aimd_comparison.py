import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.gridspec as gridspec

# plot settings
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
smallsize = 10
largesize = 10
plt.rcParams.update({"font.size": largesize})
plt.rc("xtick", labelsize=smallsize, direction="in")
plt.rc("ytick", labelsize=smallsize, direction="in")
plt.rc("axes", labelsize=largesize)
plt.rc("axes", titlesize=largesize, linewidth=0.7)
plt.rc("legend", fontsize=largesize)
plt.rc("lines", markersize=8, linewidth=2)
plt.rc(
    "legend",
    frameon=True,
    framealpha=1,
)
plt.rcParams["figure.figsize"] = [3.25, 3.25]
plt.rc("text", usetex=False)
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
matplotlib.rcParams["mathtext.default"] = "regular"
purple = "#9234eb"
green = "#3cb54a"
colors = ["#003f5c", "#374c80", "#7a5195", "#bc5090", "#ef5675", "#ff764a", "#ffa600"]

# load data
runs = [0, 1, 2, 3, 4]
seps = [6.70, 10.05, 13.40, 16.75]
data_dir = "../../../data/nnp_benchmarking/"
rdfs_all = np.load(f"{data_dir}/rdfs_all.npy", allow_pickle=True)
vdos_all = np.load(f"{data_dir}/vdos_all.npy", allow_pickle=True)
dens_all = np.load(f"{data_dir}/dens_all.npy", allow_pickle=True)
rdfs_all_err = np.load(f"{data_dir}/rdfs_all_err.npy", allow_pickle=True)
vdos_all_err = np.load(f"{data_dir}/vdos_all_err.npy", allow_pickle=True)
dens_all_err = np.load(f"{data_dir}/dens_all_err.npy", allow_pickle=True)


def plot_together(rdf, rdf_err, dens, dens_err, vdos, vdos_err, labels):

    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    gs = fig.add_gridspec(ncols=3, nrows=28)

    colors_tmp = [colors[0], "#BD041C"]
    linestyle = ["-", "--"]

    # Plot RDFs
    data = rdf
    data_err = rdf_err

    rdfs_to_plot = ["O-O", "Cl-O", "Na-O", "H-O", "Cl-H", "H-Na", "H-H"]

    count = 0
    for name, d in data[0].items():  # loop over atoms
        if name in rdfs_to_plot:
            data_tmp = [dat[name] for dat in data]
            data_err_tmp = [dat[name] for dat in data_err]

            ax0 = fig.add_subplot(gs[4 * count : 4 * count + 4, 0])

            for i, d in enumerate(data_tmp):
                x = d[0]
                y = d[1]
                y_err = data_err_tmp[i]
                ax0.plot(
                    x,
                    y,
                    label=labels[i % len(labels)],
                    lw=2,
                    linestyle=linestyle[i % len(linestyle)],
                    color=colors_tmp[i % len(colors_tmp)],
                )
                if data_err_tmp[i] != []:
                    ax0.fill_between(
                        x,
                        y - y_err,
                        y + y_err,
                        alpha=0.5,
                        color=colors_tmp[i % len(colors_tmp)],
                    )

            ax0.set_yticks([])

            if count == 0:
                ax0.set_title("RDF")

            if count != len(rdfs_to_plot) - 1:
                print("")
                ax0.set_xticklabels([])
            else:
                ax0.set_xlabel(r"r ($\mathrm{\AA{}}$)")

            if name == "H-O":
                ax0.set_ylim(0, 6)

            ax0.text(0.75, 0.8, name, transform=ax0.transAxes, fontsize=10)
            ax0.set_xlim(0.5, 6)
            ax0.grid(linestyle="--")
            ax0.tick_params(direction="in")
            count += 1

    # plot density profiles
    data = dens
    data_err = dens_err

    count = 0
    for name, d in data[0].items():  # loop over atoms
        data_tmp = [dat[name] for dat in data]
        data_err_tmp = [dat[name] for dat in data_err]

        ax0 = fig.add_subplot(gs[7 * count : 7 * count + 7, 1])

        for i, d in enumerate(data_tmp):
            x = d[0] - d[0][int(len(d[0]) / 2)]
            half_length = int(len(d[0]) / 2)
            half_y = (d[1][:half_length] + np.flip(d[1])[:half_length]) / 2
            y = np.concatenate((half_y, np.flip(half_y)))
            half_y_err = (
                data_err_tmp[i][:half_length] + np.flip(data_err_tmp[i])[:half_length]
            ) / 2
            y_err = np.concatenate((half_y_err, np.flip(half_y_err)))

            ax0.plot(
                x,
                y,
                label=labels[i % len(labels)],
                lw=2,
                linestyle=linestyle[i % len(linestyle)],
                color=colors_tmp[i % len(colors_tmp)],
            )
            if data_err_tmp[i] != []:
                ax0.fill_between(
                    x,
                    y - y_err,
                    y + y_err,
                    alpha=0.5,
                    color=colors_tmp[i % len(colors_tmp)],
                )

        ax0.set_yticks([])

        if count == 0:
            ax0.set_title("Density")
        if count != len(data[0].items()) - 1:
            # ax0.set_xticks([])
            print("")
            ax0.set_xticklabels([])
        else:
            ax0.set_xlabel(r"z ($\mathrm{\AA{}}$)")

        ax0.text(0.85, 0.8, name, transform=ax0.transAxes, fontsize=10)
        ax0.set_xlim([-4, 4])
        ax0.grid(linestyle="--")
        ax0.tick_params(direction="in")
        count += 1

    # plot VDOS
    data = vdos
    data_err = vdos_err

    count = 0
    for name, d in data[0].items():  # loop over atoms
        data_tmp = [dat[name] for dat in data]
        data_err_tmp = [dat[name] for dat in data_err]

        ax0 = fig.add_subplot(gs[7 * count : 7 * count + 7, 2])

        for i, d in enumerate(data_tmp):
            x = d[0]
            y = d[1]
            y_err = data_err_tmp[i]
            ax0.plot(
                x,
                y,
                label=labels[i % len(labels)],
                lw=2,
                linestyle=linestyle[i % len(linestyle)],
                color=colors_tmp[i % len(colors_tmp)],
            )
            if data_err_tmp[i] != []:
                ax0.fill_between(
                    x,
                    y - y_err,
                    y + y_err,
                    alpha=0.5,
                    color=colors_tmp[i % len(colors_tmp)],
                )

        if count == 0:
            ax0.set_title("VDOS")

        if count != len(data[0].items()) - 1:
            ax0.set_xticklabels([])
        else:
            ax0.set_xlabel(r"Frequency (cm$^{-1}$)")
            ax0.legend()

        ax0.text(0.85, 0.8, name, transform=ax0.transAxes, fontsize=10)
        ax0.grid(linestyle="--")
        ax0.tick_params(direction="in")

        ax0.set_xlim([0, 4500])
        ax0.set_yscale("log")
        ax0.set_yticks([])
        ax0.set_yticklabels([])

        count += 1

    plt.savefig("combined_benchmarking.pdf", bbox_inches="tight")

labels = ['NNP','AIMD']
plot_together(rdfs_all, rdfs_all_err, dens_all, dens_all_err, vdos_all, vdos_all_err, labels)