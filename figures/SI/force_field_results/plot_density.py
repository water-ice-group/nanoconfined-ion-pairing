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

# load data
runs = [0,1,2,3,4]
seps = [6.70, 10.05, 13.40, 16.75]
data_dir = '../../../data/force_field_md/confined/'
density_profiles_avg = np.load(f'{data_dir}/density_profiles.npy', allow_pickle=True)
density_profiles_err = np.load(f'{data_dir}/density_profiles_err.npy', allow_pickle=True)

def plot_density(data, data_err):
    line_types = {'Na':'solid',
                    'Cl': 'solid',
                    'O': 'solid'}

    labels = {'Na':'Na$^+$',
                    'Cl': 'Cl$^-$',
                    'O': 'H$_2$O'}

    c = {'Na':green,
                    'Cl':purple,
                    'O': 'k'}

    alpha = {'Na':1,
                    'Cl':1,
                    'O': 1}

    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=[7,6.5])
    ax = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    for j, system in enumerate(data): # loop over all seps

        c_position_l = -seps[j]/2
        c_position_r = seps[j]/2

        ax1 = ax[j]

        color = 'k'
        ax1.set_xlabel('z ($\AA$)')
        
        ax2 = ax1.twinx()  

        color = 'k'

        for i, d in system.items():
            x = d[0]
            y = d[1]
            y_err = data_err[j][i]
            if i == 'O':
                ax1.plot(x, y,linestyle=line_types[i],lw=1.5, color = c[i],label=labels[i], alpha=alpha[i])
                ax1.fill_between(x, y-y_err, y+y_err, alpha = 0.5, color = c[i])
            else:
                ax2.plot(x, 1000*y,linestyle=line_types[i],lw=1.5, color = c[i],label=labels[i], alpha=alpha[i])
                ax2.fill_between(x, 1000*(y-y_err), 1000*(y+y_err), alpha = 0.5, color = c[i])

        ax1.set_xlim(c_position_l, c_position_r)

        if j == 0 or j==2:
            ax1.set_ylabel('Water Density (molecules/$\AA^3$)', color=color)
            ax2.set_yticklabels([])
        elif j == 3 or j == 1:
            ax2.set_ylabel('Ion Density (10$^{-3}$ molecules/$\AA^3$)', color=color)  
            ax1.set_yticklabels([])
            # ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
        else:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
            
        ax1.set_yticks(np.linspace(0, 0.16, 5))
        ax2.set_yticks(np.linspace(0, 0.006*1000, 5))
        
        ax1.set_ylim(0,0.16)
        ax2.set_ylim(0,0.006*1000)
        ax1.grid(linestyle='--')
        ax2.grid(linestyle='--')    

        # if j == 0:
        #     fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, framealpha=1)


    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    plt.savefig(f'density_profiles.png', format='png', dpi=600, bbox_inches='tight', transparent=True)

plot_density(density_profiles_avg, density_profiles_err)