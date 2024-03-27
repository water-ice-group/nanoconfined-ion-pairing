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
coordination_bins = np.load(f'{data_dir}/coordination_bins.npy', allow_pickle=True)
coordination = np.load(f'{data_dir}/coordination_numbers.npy', allow_pickle=True)
coordination_err = np.load(f'{data_dir}/coordination_numbers_err.npy', allow_pickle=True)
nao_vs_r = np.mean(coordination[:,:,0],axis=1)
nao_vs_r_err = np.mean(coordination_err[:,:,0],axis=1)
clh_vs_r = np.mean(coordination[:,:,1],axis=1)
clh_vs_r_err = np.mean(coordination_err[:,:,1],axis=1)

data_dir = '../../../data/nnp_md/bulk/'
coordination_bulk = np.load(f'{data_dir}coordination_numbers.npy', allow_pickle=True)
coordination_err_bulk = np.load(f'{data_dir}/coordination_numbers_err.npy', allow_pickle=True)
coordination_bins_bulk = np.load(f'{data_dir}/coordination_bins.npy', allow_pickle=True)
nao_vs_r_bulk = np.mean(coordination_bulk[:,0])
nao_vs_r_err_bulk = np.mean(coordination_err_bulk[:,0])
clh_vs_r_bulk = np.mean(coordination_bulk[:,1])
clh_vs_r_err_bulk = np.mean(coordination_err_bulk[:,1])

fig = plt.figure(figsize=(6.7,4), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=6)
ax0 = fig.add_subplot(gs[0:4,0])
ax1 = fig.add_subplot(gs[0:4,1])
axes = [ax0, ax1]

j_values = np.arange(len(seps))
for index, j in enumerate(j_values):
    # ax = axes[index]
    
    x = coordination_bins[j][:-1]
    y1 = nao_vs_r[j]
    y2 = clh_vs_r[j]
    y1_err = nao_vs_r_err[j]
    y2_err = clh_vs_r_err[j]

    ax0.plot(x, y1, color=colors[2*j], label=f'H = {seps[j]} $\AA$')
    ax1.plot(x, y2, color=colors[2*j], label=f'H = {seps[j]} $\AA$')
        
    # if j == 0:
    #     ax.axvline(5.25,color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(third_trough,color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(cip_min[j],color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(cip_max[j],color='k',linestyle='--',alpha=0.5,lw=1)
    #     ax.axvline(ssip_min[j],color='k',linestyle='--',alpha=0.5,lw=1)
    # else:
    #     ax.axvline(cip_max[j],color='k',linestyle='--',alpha=0.5,lw=1)

x = coordination_bins_bulk[:-1]
y1 = nao_vs_r_bulk
y2 = clh_vs_r_bulk
y1_err = nao_vs_r_err_bulk
y2_err = clh_vs_r_err_bulk

ax0.plot(x, y1,'--',  color='gray', label='Bulk')
ax1.plot(x, y2, '--', color='gray', label='Bulk')

ax0.set_xlim(2.4,10)
ax0.set_ylim(3,6.8)
ax1.set_xlim(2.4,10)
ax1.set_ylim(3,6.8)

ax0.set_ylabel('Na-O Coordination number') 
ax1.set_ylabel('Cl-H Coordination number') 
ax0.set_xlabel('r$_\mathrm{Na-Cl}$ ($\AA$)')
ax1.set_xlabel('r$_\mathrm{Na-Cl}$ ($\AA$)')

ax1.legend(frameon=True,framealpha=1,edgecolor='k')
ax0.grid(linestyle='--')
ax1.grid(linestyle='--')

plt.savefig(f'coordination.pdf', format='pdf', bbox_inches='tight')