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
coordination_bins = np.load(f'{data_dir}/coordination_bins.npy', allow_pickle=True)
coordination = np.load(f'{data_dir}/coordination_numbers.npy', allow_pickle=True)
coordination_err = np.load(f'{data_dir}/coordination_numbers_err.npy', allow_pickle=True)
nao_vs_r = np.mean(coordination[:,:,0],axis=1)
nao_vs_r_err = np.mean(coordination_err[:,:,0],axis=1)
clh_vs_r = np.mean(coordination[:,:,1],axis=1)
clh_vs_r_err = np.mean(coordination_err[:,:,1],axis=1)

j_values = [0, len(seps)-1]
for index, j in enumerate(j_values):
    fig = plt.figure()
    ax = plt.gca()
    
    x = coordination_bins[j][:-1]
    y1 = nao_vs_r[j]
    y2 = clh_vs_r[j]
    y1_err = nao_vs_r_err[j]
    y2_err = clh_vs_r_err[j]

    ax.plot(x, y1, color=purple, label='Na-O')
    ax.plot(x, y2, color=green, label='Cl-H')
    ax.fill_between(x, y1-y1_err, y1+y1_err, alpha=0.2,color=purple)
    ax.fill_between(x, y2-y2_err, y2+y2_err, alpha=0.2, color=green)

    ax.set_xlim(2.4,10)
    ax.set_ylim(3,7)

    ax.set_xlabel('r$_\mathrm{Na-Cl}$ ($\AA$)')
    
    ax.set_ylabel('Coordination number') 

    leg = ax.legend(loc='lower right')
    leg.get_frame().set_edgecolor('k')

    # if j == 0:
    #     ax.axvline(5.25,color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(third_trough,color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(cip_min[j],color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(cip_max[j],color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(ssip_min[j],color='k',linestyle='--',alpha=0.5,lw=1)
    # else:
    #     ax.axvline(cip_max[j],color='k',linestyle='--',alpha=0.5,lw=1)

    plt.savefig(f'coordination_{j}.pdf', format='pdf', bbox_inches='tight')