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
data_dir = "../../../data/force_field_md/confined/"
pmf_bins_L45 = np.load(f'{data_dir}/pmf_bins.npy', allow_pickle=True)
pmf_L45 = np.load(f'{data_dir}/pmf_avg.npy', allow_pickle=True)
pmf_err_L45 = np.load(f'{data_dir}/pmf_err.npy', allow_pickle=True)
pmf_bins_large = np.load(f'{data_dir}/pmf_L90_bins.npy', allow_pickle=True)
pmfs_large = np.load(f'{data_dir}/pmf_L90.npy', allow_pickle=True)
pmf_err_large = np.load(f'{data_dir}/pmf_L90_err.npy', allow_pickle=True)

plt.figure(figsize=(7,3.25), constrained_layout=True)
gs = gridspec.GridSpec(1,4,wspace=0.1)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharey=ax1)
ax3 = plt.subplot(gs[2], sharey=ax1)
ax4 = plt.subplot(gs[3], sharey=ax1)

axes = [ax1, ax2, ax3, ax4]
fig = plt.gcf()

for j, sep in enumerate(seps):
    ax = axes[j]
    
    ax.plot(pmf_bins_L45[j], pmf_L45[j], color=colors[0], label=f'L $\\approx$ 45 $\AA$')
    ax.fill_between(pmf_bins_L45[j], pmf_L45[j]+pmf_err_L45[j],pmf_L45[j]-pmf_err_L45[j], color=colors[0], alpha=0.5)

    ax.plot(pmf_bins_large[0], pmfs_large[j],'--', color=colors[4], label=f'L $\\approx$ 90 $\AA$')

    if j != 0:
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set_ylabel('Free energy ($k_\mathrm{B}T$)')

    if j == 3:
        ax.legend(bbox_to_anchor=(1.04,-0.02), loc='lower right')
    
    ax.set_xlim(2,10)
    ax.set_ylim(-5,5)
    ax.grid('both',linestyle='--')
    ax.set_title(f'H = {seps[j]} $\AA$')

    ax.set_xlabel('r ($\AA$)')
    
plt.tight_layout()
plt.savefig("finite_size_test.pdf", bbox_inches='tight')