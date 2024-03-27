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
lime = '#93C572'
colors = ['#003f5c','#374c80','#7a5195','#bc5090','#ef5675','#ff764a','#ffa600']

# load data
runs = [0,1,2,3,4]
seps = [6.70, 10.05, 13.40, 16.75]
data_dir = '../../data/nnp_md/confined/'
pmf_bins = np.load(f'{data_dir}/pmf_bins.npy', allow_pickle=True)
pmf_avg = np.load(f'{data_dir}/pmf_avg.npy', allow_pickle=True)
pmf_err = np.load(f'{data_dir}/pmf_err.npy', allow_pickle=True)

data_dir = '../../data/nnp_md/implicit_carbon/'
pmf_implicit = np.load(f'{data_dir}/carbone_pmf_avg.npy', allow_pickle=True)[0]
pmf_implicit_bins = np.load(f'{data_dir}/carbone_pmf_bins.npy', allow_pickle=True)[0]
pmf_implicit_err = np.load(f'{data_dir}/carbone_pmf_err.npy', allow_pickle=True)[0]

kcal_per_mol_per_kT = 0.596

plt.figure(figsize=[2.6,3.25])

for j, sep in reversed(list(enumerate(seps))):
    if j == 0:
        plt.plot(pmf_bins[j], pmf_avg[j], color=colors[j*2], label=f'Explicit carbon')
        plt.fill_between(pmf_bins[j], pmf_avg[j]+pmf_err[j],pmf_avg[j]-pmf_err[j], color=colors[j*2], alpha=0.3)

        plt.plot(pmf_implicit_bins, pmf_implicit, color=lime, label='Implicit carbon')
        plt.fill_between(pmf_implicit_bins, pmf_implicit+pmf_implicit_err, pmf_implicit-pmf_implicit_err, color=lime, alpha=0.3)

plt.xlim(2,10)
plt.ylim(-3,4)
plt.grid('both',linestyle='--')

plt.xlabel('r$_\mathrm{Na-Cl}$ ($\AA$)')
plt.ylabel('Free Energy ($k_\mathrm{B}T$)')

kcal_per_mol_per_kT = 0.596
ax2 = plt.gca().twinx()
ax2.set_ylim(-3 * kcal_per_mol_per_kT, 4 * kcal_per_mol_per_kT)
ax2.set_ylabel('Free Energy (kcal/mol)')

plt.savefig('pmf_implicit.png', format='png', dpi=600, bbox_inches='tight')