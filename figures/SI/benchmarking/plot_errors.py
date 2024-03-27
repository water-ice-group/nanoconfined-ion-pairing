import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.gridspec as gridspec

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
data_dir = '../../../data/nnp_benchmarking/'
dft_forces_all = np.load(f'{data_dir}/dft_forces.npy', allow_pickle=True)
nnp_forces_all = np.load(f'{data_dir}/nnp_forces.npy', allow_pickle=True)
energy_rmses = np.load(f'{data_dir}/energy_rmses.npy', allow_pickle=True)
force_err_vector = np.load(f'{data_dir}/force_err.npy', allow_pickle=True)
forces_all = np.load(f'{data_dir}/force_err_all.npy', allow_pickle=True)

plt.figure(figsize=(7,3.25), constrained_layout=True)
gs = gridspec.GridSpec(1,2)
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[0])
fig = plt.gcf()

all_atom_types = ['C','O','H','Na','Cl']
atom_types_all = np.load(f'{data_dir}/atom_types_all.npy', allow_pickle=True)
n_calculations = 100

ax1.text(-0.2, 1.0, 'B', transform=ax1.transAxes,
      fontsize=10, fontweight='bold')

ax2.text(-0.2, 1.0, 'A', transform=ax2.transAxes,
      fontsize=10, fontweight='bold')

ax1.set_axisbelow(True)
ax1.grid(linestyle='--')
bar_width = 0.15
indices = np.arange(len(all_atom_types))#+1) 

for i, sep in enumerate(seps):
    force_err = list(np.mean(forces_all[i],axis=1)) 
    # print(force_err)
    color = colors[i*2%len(colors)]
    ax1.set_xlabel('Atom type')
    ax1.set_ylabel('Force RMSE (meV/$\AA$)')
    ax1.bar(indices + bar_width*i, force_err, bar_width,label=f'H = {seps[i]}$\AA$',
           color=matplotlib.colors.to_rgba(color,0.8),edgecolor=color,lw=1)

ax1.set_xticks(indices + bar_width*1.5)
ax1.set_xticklabels(list(all_atom_types))# + ['Avg.'])
ax1.set_ylim(0,85)
        
ax1.legend(loc=0)

ax2.set_axisbelow(True)
indices = np.arange(len(seps))
ax2.grid(linestyle='--')
bar_width = 0.35

color = colors[0]
ax2.set_xlabel('Indices')
ax2.set_ylabel('Energy RMSE (meV/atom)')
ax2.bar(indices - bar_width/2, energy_rmses, bar_width, color=color, label='Energy RMSE')
ax2.tick_params(axis='y')

ax3 = ax2.twinx()  

color = '#A0DA39'
color='#80AF2E'
color=colors[0]
ax3.set_ylabel('Force RMSE (meV/$\AA$)')
ax3.bar(indices + bar_width/2, force_err_vector, bar_width, label='Force RMSE', ec=color, color=matplotlib.colors.to_rgba(color,0.5))
ax3.tick_params(axis='y')

ax2.set_xticks(indices)
ax2.set_xticklabels(seps)

ax2.set_xlabel('Slit Height ($\AA$)')

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax3.legend(lines + lines2, labels + labels2, loc='upper left')
ax3.set_ylim(0,45)
ax2.set_ylim(0,0.38)

plt.tight_layout()

plt.savefig("errors_bargraph.pdf", bbox_inches='tight')

#######################

markers = ['o','s','^','>']
plt.figure(figsize=(7,7.5))
gs = gridspec.GridSpec(3, 4)
ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[0,2:])
ax3 = plt.subplot(gs[1,0:2])
ax4 = plt.subplot(gs[1,2:])
ax5 = plt.subplot(gs[2,1:3])
fig = plt.gcf()
# gs.tight_layout(fig)
axes = [ax1,ax2,ax3,ax4,ax5]
for i in range(len(all_atom_types)):
    ax = axes[i]
    for j, sep in reversed(list(enumerate(seps))):
        
        if all_atom_types[i] in atom_types_all[j]:
            # ax.set_title(all_atom_types[i])
            index = atom_types_all[j].index(all_atom_types[i])
            for k in range(3):
                for l in range(n_calculations):
                    if l == 0 and k == 0:
                        ax.plot(dft_forces_all[j][l,index][:,k],nnp_forces_all[j][l,index][:,k],linestyle='None',
                                 marker=markers[j],markerfacecolor=matplotlib.colors.to_rgba(colors[j*2],0.5),
                                 markeredgecolor=matplotlib.colors.to_rgba(colors[j*2],0.8),label=f'H = {seps[j]} $\AA$')
                    else:
                        ax.plot(dft_forces_all[j][l,index][:,k],nnp_forces_all[j][l,index][:,k],linestyle='None',
                                 marker=markers[j],markerfacecolor=matplotlib.colors.to_rgba(colors[j*2],0.5),
                               markeredgecolor=matplotlib.colors.to_rgba(colors[j*2],0.8))
            
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    parity = np.linspace(-10000,10000)
    ax.plot(parity,parity,'k-',alpha=0.5)
    
    ax.text(0.05,0.9,all_atom_types[i],transform=ax.transAxes)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid('both',linestyle='--')
    ax.set_xlabel('F$_\mathrm{DFT}$ (meV/$\AA$)')
    ax.set_ylabel('F$_\mathrm{NNP}$ (meV/$\AA$)')
    if i == 1:
        ax.legend(labelspacing=0.1,handletextpad=0.05, bbox_to_anchor=(1.04,-0.02), loc='lower right')
    fig.tight_layout()
plt.savefig("force_errors.png", format='png', dpi=300, bbox_inches='tight')