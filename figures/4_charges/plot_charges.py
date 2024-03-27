import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
data_dir = '../../data/charge_analysis/'
c_charge_bins = np.load(f'{data_dir}/bader_c_charge_bins.npy', allow_pickle=True)
c_charge_mean = np.load(f'{data_dir}/bader_c_charge_mean.npy', allow_pickle=True)
c_charge_err = np.load(f'{data_dir}/bader_c_charge_err.npy', allow_pickle=True)
c_charges_all = np.load(f'{data_dir}/bader_c_charges.npy', allow_pickle=True)

for j in range(len(seps)):
    if j == 0:

        plt.figure(figsize=(3,3))

        # sodium ions
        plt.plot(c_charge_bins[j][0], c_charge_mean[j][0], '-', color=purple,label='r$_\parallel^\mathrm{Na}$')
        plt.fill_between(c_charge_bins[j][0], c_charge_mean[j][0] + c_charge_err[j][0], 
                        c_charge_mean[j][0] - c_charge_err[j][0], alpha=0.3, color=purple)

        # chloride ions
        plt.plot(c_charge_bins[j][1], c_charge_mean[j][1], '-', color=green, label='r$_\parallel^\mathrm{Cl}$')
        plt.fill_between(c_charge_bins[j][1], c_charge_mean[j][1] + c_charge_err[j][1], 
                        c_charge_mean[j][1] - c_charge_err[j][1], alpha=0.3, color=green)

        plt.hlines(np.mean(c_charges_all[j,:,:]), 0, 9, linestyle='--', color='k')

        plt.xlabel('r$_\parallel$ ($\AA$)')
        plt.ylabel('Carbon charge (e)')
        plt.xlim(0,8)
        plt.legend(frameon=True, framealpha=1, edgecolor='k')
        
        plt.savefig(f'bader_carbon_charge_vs_r_sep{seps[j]}.pdf', format='pdf', bbox_inches='tight')

######### charge visualization #########
def save_ion_coords(path, lx=12.350, ly=12.834):
    fn = path + 'structure.pdb'
    x_na = []
    y_na = []
    x_cl = []
    y_cl = []
    with open(fn, 'r') as file1:
        lines = file1.readlines()
        for line in lines:
            if "ATOM" in line:
                columns = line.split()
                if columns[2] == 'Na':
                    x_na.append(float(columns[3])%lx)
                    y_na.append(float(columns[4])%ly)
                elif columns[2] == 'Cl':
                    x_cl.append(float(columns[3])%lx)
                    y_cl.append(float(columns[4])%ly)
    return np.array((x_na,y_na)), np.array((x_cl,y_cl))

def save_carbon_coords(path):
    fn = path + 'structure.pdb'
    x_pos = []
    y_pos = []
    z_pos = []
    with open(fn, 'r') as file1:
        lines = file1.readlines()
        for line in lines:
            if "ATOM" in line:
                columns = line.split()
                if columns[2] == 'C':
                    x_pos.append(float(columns[3]))
                    y_pos.append(float(columns[4]))
                    z_pos.append(float(columns[5]))
    return np.array((x_pos,y_pos,z_pos))

charges = np.loadtxt(f'{data_dir}/bader_charges.txt')
pos = save_carbon_coords(data_dir)
pos_na, pos_cl = save_ion_coords(data_dir)

c_size = 170
cl_size = 181
na_size = 95
c_markersize = 415
na_markersize = c_markersize*na_size/c_size
cl_markersize = c_markersize*cl_size/c_size

# plot the carbon positions
bottom_layer = np.where(pos[2,:]==7.5)

x_shift = 0.0623 # shift axes for visualization purposes
y_shift = -0.25005

fig = plt.figure(figsize=(3.1,5.5))
ax = plt.axes()
norm = mcolors.TwoSlopeNorm(vcenter=0,vmin=-0.05, vmax=0.05)
im = ax.scatter(pos[0,bottom_layer]-x_shift, pos[1,bottom_layer]-y_shift, s=c_markersize, ec='k', c=charges[bottom_layer], cmap='seismic', norm=norm)


# plt.yticks([])
# plt.xticks([])
plt.ylabel('y ($\AA$)')
plt.xlabel('x ($\AA$)')
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position("top")

print(ax.get_xlim(),ax.get_ylim())
plt.xticks(np.arange(0, ax.get_xlim()[1] - ax.get_xlim()[0], 2.0))
plt.yticks(np.arange(0, ax.get_ylim()[1] - ax.get_ylim()[0], 2.0))
locs, labels = plt.xticks() 
print(locs, labels)

v = np.linspace(-0.05, 0.05, 3)
cbar = fig.colorbar(im, ax=ax, label='Charge (e)', orientation="horizontal", pad=0.13, ticks=v)

ax.scatter(pos_na[0]-x_shift, pos_na[1]-y_shift, s=na_markersize*2, ec='k', c=purple)
ax.scatter(pos_cl[0]-x_shift, pos_cl[1]-y_shift, s=cl_markersize*2, ec='k', c=green)

ax.set_aspect('equal', adjustable='box')

plt.savefig('charge_pic.pdf', bbox_inches='tight', format='pdf')