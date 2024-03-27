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
data_dir = '../../data/nnp_md/confined/'
pmf_bins = np.load(f'{data_dir}/pmf_bins.npy', allow_pickle=True)
pmf_avg = np.load(f'{data_dir}/pmf_avg.npy', allow_pickle=True)
pmf_err = np.load(f'{data_dir}/pmf_err.npy', allow_pickle=True)

data_dir = '../../data/nnp_md/bulk/'
rdf_bulk = np.load(f'{data_dir}/rdf.npy')
rdf_bulk_bins = np.load(f'{data_dir}/rdf_bins.npy')

kcal_per_mol_per_kT = 0.596

for j, sep in reversed(list(enumerate(seps))):
    plt.plot(pmf_bins[j], pmf_avg[j], color=colors[j*2], label=f'H = {seps[j]:.2f} $\AA$')
    plt.fill_between(pmf_bins[j], pmf_avg[j]+pmf_err[j],pmf_avg[j]-pmf_err[j], color=colors[j*2], alpha=0.3)

plt.plot(rdf_bulk_bins,-np.log(rdf_bulk),linestyle='dotted',color='k',alpha=0.75,label='Bulk')

plt.xlim(2,10)
plt.ylim(-3,4)
plt.grid('both',linestyle='--')

plt.xlabel('r$_\mathrm{Na-Cl}$ ($\AA$)')
plt.ylabel('Free Energy ($k_\mathrm{B}T$)')
leg = plt.legend(frameon=True,framealpha=1)
leg.get_frame().set_edgecolor('k')

ax2 = plt.gca().twinx()
ax2.set_ylim(-3 * kcal_per_mol_per_kT, 4 * kcal_per_mol_per_kT)
ax2.set_ylabel('Free Energy (kcal/mol)')

plt.savefig('pmf.png', format='png', dpi=600, bbox_inches='tight')