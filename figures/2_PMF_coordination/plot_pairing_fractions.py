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
lime = '#93C572'
colors = ['#003f5c','#374c80','#7a5195','#bc5090','#ef5675','#ff764a','#ffa600']

# load data
runs = [0,1,2,3,4]
seps = [6.70, 10.05, 13.40, 16.75]
data_dir = '../../data/nnp_md/confined/'
ion_pairing_avg = np.load(f'{data_dir}/ion_pairing_fracs.npy', allow_pickle=True)
ion_pairing_err = np.load(f'{data_dir}/ion_pairing_fracs_err.npy', allow_pickle=True)

data_dir = '../../data/nnp_md/bulk/'
ion_pairing_avg_bulk = np.load(f'{data_dir}/ion_pairing_bulk.npy', allow_pickle=True)
ion_pairing_err_bulk = np.load(f'{data_dir}/ion_pairing_bulk_err.npy', allow_pickle=True)

color1 = lime
color2 = colors[1]
color3 = colors[5]

plt.errorbar(seps[:],ion_pairing_avg[:,0], yerr = ion_pairing_err[:,0], marker='None', color='k', capsize=3, linestyle='None')
plt.plot(seps,ion_pairing_avg[:,0],'o-',label='CIP',markersize=7, color=color1, markerfacecolor=matplotlib.colors.to_rgba(color1,0.7),markeredgecolor=color1,markeredgewidth=1)
plt.errorbar(seps[:],ion_pairing_avg[:,1], yerr = ion_pairing_err[:,1], marker='None', color='k', capsize=3, linestyle='None')
plt.plot(seps,ion_pairing_avg[:,1],'s-',label='SSIP',markersize=7, color=color2, markerfacecolor=matplotlib.colors.to_rgba(color2,0.7),markeredgecolor=color2,markeredgewidth=1)
plt.errorbar(seps[:],ion_pairing_avg[:,2], yerr = ion_pairing_err[:,2], marker='None', color='k', capsize=3, linestyle='None')
plt.plot(seps,ion_pairing_avg[:,2],'d-',label='Free',markersize=7, color=color3, markerfacecolor=matplotlib.colors.to_rgba(color3,0.7),markeredgecolor=color3,markeredgewidth=1)

# bulk values
plt.errorbar([18],ion_pairing_avg_bulk[0], yerr = ion_pairing_err_bulk[0], marker='None', color='k', capsize=3, linestyle='None')
plt.errorbar([18],ion_pairing_avg_bulk[2], yerr = ion_pairing_err_bulk[2], marker='None', color='k', capsize=3, linestyle='None')
plt.errorbar([18],ion_pairing_avg_bulk[1], yerr = ion_pairing_err_bulk[1], marker='None', color='k', capsize=3, linestyle='None')

plt.plot([18],ion_pairing_avg_bulk[0],marker='o',#markerfacecolor='None',
                    color=color1,markersize=7,markeredgecolor='k')
plt.plot([18],ion_pairing_avg_bulk[1],marker='s',#markerfacecolor='None',
                    color=color2,markersize=7,markeredgecolor='k')
plt.plot([18],ion_pairing_avg_bulk[2],marker='d',#markerfacecolor='None',
                    color=color3,markersize=7,markeredgecolor='k')

plt.xticks(np.arange(0, 21, step=3,))
x = plt.xticks()[0]
new_labels = [str(label) if label != 18 else 'Bulk' for label in x]
plt.xticks(x, new_labels)
plt.xlim(6.2,18.5)

plt.xlabel('Slit height ($\AA$)')
plt.ylabel('Fraction of Na$^+$')
plt.grid('both',linestyle='--')
plt.savefig('frac_ion_pairs.png', format='png', dpi=600, bbox_inches='tight')