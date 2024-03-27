import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plot settings
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
smallsize = 10
largesize = 10
plt.rcParams.update({'font.size': largesize})
plt.rc('xtick', labelsize = smallsize, direction='in')
plt.rc('ytick', labelsize= smallsize, direction='in')
plt.rc('axes', labelsize = largesize)
plt.rc('axes', titlesize = largesize, linewidth=0.7)
plt.rc('legend', fontsize=largesize)
plt.rc('lines', markersize=8, linewidth=2)
plt.rc('legend', frameon=True,framealpha=1,)
plt.rcParams['figure.figsize'] = [3.25,3.25]
plt.rc('text', usetex=False)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['mathtext.default'] = 'regular'
purple = '#9234eb'
green = '#3cb54a'
colors = ['#003f5c','#374c80','#7a5195','#bc5090','#ef5675','#ff764a','#ffa600']

# load data
runs = [0,1,2,3,4]
seps = [6.70, 10.05, 13.40, 16.75]
data_dir = '../../../data/nnp_md/confined/'
ion_pair_acfs = np.load(f'{data_dir}/ion_pair_acf.npy', allow_pickle=True)
ion_pair_acfs_err = np.load(f'{data_dir}/ion_pair_acf_err.npy', allow_pickle=True)
ion_pair_residence_times = np.load(f'{data_dir}/ion_pair_residence_times.npy', allow_pickle=True)
ion_pair_residence_times_err = np.load(f'{data_dir}/ion_pair_residence_times_err.npy', allow_pickle=True)
times_all = np.load(f'{data_dir}/times.npy', allow_pickle=True)

fig = plt.figure(figsize=(6,3.25))
gs = fig.add_gridspec(ncols=2, nrows=1)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])

for j, sep in enumerate(seps):
    ax0.plot(times_all[j,0][:len(ion_pair_acfs[j])], ion_pair_acfs[j][:len(times_all[j,0])],color=colors[2*j], label=f'H = {seps[j]} $\AA$')
    ax0.fill_between(times_all[j,0][:len(ion_pair_acfs[j])], ion_pair_acfs[j][:len(times_all[j,0])] + ion_pair_acfs_err[j][:len(times_all[j,0])], ion_pair_acfs[j][:len(times_all[j,0])] - ion_pair_acfs_err[j][:len(times_all[j,0])], alpha=0.5,color=colors[2*j])

color='k'
ax1.errorbar(seps,ion_pair_residence_times,yerr=ion_pair_residence_times_err, marker='None', color='k', capsize=3, linestyle='None')
ax1.plot(seps,ion_pair_residence_times,'s-', color=color, markerfacecolor=matplotlib.colors.to_rgba(color,0.5),markeredgecolor=color,markeredgewidth=1)

ax1.set_xlabel(f'Slit height ($\AA$)')
ax1.set_ylabel('Ion pair residence time (ps)')
ax1.grid(linestyle='--')
ax0.set_xscale('log')
ax0.set_xlim(0.5,2000)
ax0.set_ylim(0,1)
ax0.set_xlabel('Time (ps)')
ax0.set_ylabel('Ion pair lifetime ACF')
ax0.grid(linestyle='--')
ax0.legend(frameon=True,framealpha=1, edgecolor='k')
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('residence_times.png', format='png', dpi=600, bbox_inches='tight')