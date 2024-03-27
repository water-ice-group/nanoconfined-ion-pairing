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
data_dir = '../../../data/charge_analysis/'
ion_charge_bins = np.load(f'{data_dir}/bader_ion_charge_bins.npy', allow_pickle=True)
ion_charge_mean = np.load(f'{data_dir}/bader_ion_charge_mean.npy', allow_pickle=True)
ion_charge_err = np.load(f'{data_dir}/bader_ion_charge_err.npy', allow_pickle=True)

plt.figure()
for j in range(len(seps)):
    # sodium ions
    plt.plot(ion_charge_bins[j][0], ion_charge_mean[j][0], '-', color=colors[2*j],label=f'H = {seps[j]} $\AA$')
    plt.fill_between(ion_charge_bins[j][0], ion_charge_mean[j][0] + ion_charge_err[j][0], 
                     ion_charge_mean[j][0] - ion_charge_err[j][0], alpha=0.5, color=colors[2*j])

    #  # chloride ions
    plt.plot(ion_charge_bins[j][1], -ion_charge_mean[j][1], '-', color=colors[2*j])
    plt.fill_between(ion_charge_bins[j][1], -ion_charge_mean[j][1] + ion_charge_err[j][1], 
                     -ion_charge_mean[j][1] - ion_charge_err[j][1], alpha=0.5, color=colors[2*j])

    plt.xlabel('z ($\AA$)')
    plt.ylabel('Ion charge magnitude (e)')
    plt.grid(linestyle='--')
    leg = plt.legend(frameon=True, framealpha=1, edgecolor='k', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0.68,0.97)

    plt.text(4.1, 0.94, 'Na$^+$', color=purple, fontsize=11, bbox=dict(facecolor='white', edgecolor='k',boxstyle='round, rounding_size=0.5'))
    plt.text(4.1, 0.79, 'Cl$^-$', color=green, fontsize=11, bbox=dict(facecolor='white', edgecolor='k',boxstyle='round, rounding_size=0.5'))

    plt.savefig(f'bader_ion_charges_vs_z.pdf', format='pdf', bbox_inches='tight')

c_charge_bins = np.load(f'{data_dir}/bader_c_charge_bins.npy', allow_pickle=True)
c_charge_mean = np.load(f'{data_dir}/bader_c_charge_mean.npy', allow_pickle=True)
c_charge_err = np.load(f'{data_dir}/bader_c_charge_err.npy', allow_pickle=True)
c_charges_all = np.load(f'{data_dir}/bader_c_charges.npy', allow_pickle=True)

fig = plt.figure(figsize=(6,3.25))
gs = fig.add_gridspec(ncols=2, nrows=1)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])

for j in range(len(seps)):

    # sodium ions
    ax0.plot(c_charge_bins[j][0], c_charge_mean[j][0], color=colors[2*j], markersize=4,
             alpha=1, label=f'H = {seps[j]} $\AA$')

     # chloride ions
    ax1.plot(c_charge_bins[j][1], c_charge_mean[j][1], color=colors[2*j])
    ax0.hlines(np.mean(c_charges_all[j,:,:]), 0, 9, linestyle='--', color='k')
    ax1.hlines(np.mean(c_charges_all[j,:,:]), 0, 9, linestyle='--', color='k')

    ax0.set_xlabel('r$_{Na}$ ($\AA$)')
    ax1.set_xlabel('r$_{Cl}$ ($\AA$)')
    ax0.set_ylabel('Carbon charge (e)')
    ax1.set_ylabel('Carbon charge (e)')
    ax0.legend(frameon=True, framealpha=1, edgecolor='k')
    ax0.grid(linestyle='--')
    ax1.grid(linestyle='--')
plt.tight_layout()

plt.savefig(f'bader_carbon_charge_vs_r_all.pdf', format='pdf', bbox_inches='tight')

c_charge_bins_mull = np.load(f'{data_dir}/mulliken_c_charge_bins.npy', allow_pickle=True)
c_charge_mean_mull = np.load(f'{data_dir}/mulliken_c_charge_mean.npy', allow_pickle=True)
c_charge_err_mull = np.load(f'{data_dir}/mulliken_c_charge_err.npy', allow_pickle=True)
c_charges_all_mull = np.load(f'{data_dir}/mulliken_c_charges.npy', allow_pickle=True)

fig, ax = plt.subplots()

# Mulliken charges
j = 0
i = 0
x = c_charge_bins_mull[j][i]
y = c_charge_mean_mull[j][i]
yerr = c_charge_err_mull[j][i]
ax.plot(x, y, '-', color=purple,label='r$_\mathrm{Na}$')
ax.fill_between(x, y + yerr, y - yerr, alpha=0.3, color=purple)
i = 1
x = c_charge_bins_mull[j][i]
y = c_charge_mean_mull[j][i]
yerr = c_charge_err_mull[j][i]
ax.plot(x, y, '-', color=green, label='r$_\mathrm{Cl}$')
ax.fill_between(x, y + yerr, y - yerr, alpha=0.3, color=green)
ax.hlines(np.mean(c_charges_all_mull[j,:,:]), 0, 9, linestyle='--', color='k')

ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('Carbon charge (e)')
ax.set_xlim(0,8)

plt.tight_layout()
        
plt.savefig(f'carbon_charge_vs_r_mulliken.pdf', format='pdf', bbox_inches='tight')