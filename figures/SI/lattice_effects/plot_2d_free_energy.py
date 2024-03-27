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
H_shifted = np.load(f'{data_dir}/2d_density.npy', allow_pickle=True)
carbon_positions_split = np.load(f'{data_dir}/carbon_positions_split.npy', allow_pickle=True)
xedges, yedges, xmin, ymin = np.load(f'{data_dir}/2d_density_extent.npy', allow_pickle=True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6.15,3.25), constrained_layout=True)

titles = ['Water','Cl$^-$', 'Na$^+$']

# for colorbar, get max value from all three particle types
max_value = np.max(H_shifted)

for i, ax in enumerate(axes.flat):
    ax.plot(carbon_positions_split[0,0][:,0],carbon_positions_split[0,0][:,1],'ko',alpha=0.5,markersize=25,markerfacecolor='None')

    # ax.set_title(titles[i])
    
    H = H_shifted[i]
    im = ax.imshow(H, interpolation='gaussian', origin='lower',extent=[xedges[0]-xmin, xedges[-1]-xmin, 
                                                                  yedges[0]-ymin, yedges[-1]-ymin],
             vmin=0,vmax=max_value)
       
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x ($\AA$)')
    
    if i != 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('y ($\AA$)')

cbar_ax = fig.add_axes([1.01, 0.25, 0.02, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Free energy (k$_\mathrm{B}$T)', rotation=270, labelpad=15) 
    
plt.savefig('2d_density.pdf', format='pdf', bbox_inches='tight')