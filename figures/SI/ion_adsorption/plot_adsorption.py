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
adsorption = np.load(f'{data_dir}/adsorption.npy', allow_pickle=True)
adsorption_err = np.load(f'{data_dir}/adsorption_err.npy', allow_pickle=True)

color1 = purple
color2 = green

plt.errorbar(seps[:],adsorption[0,:], yerr = adsorption_err[0,:], marker='None', color='k', capsize=3, linestyle='None')
plt.plot(seps,adsorption[0,:],'o-',label='Na$^+$',markersize=7, color=color1, markerfacecolor=matplotlib.colors.to_rgba(color1,0.7),markeredgecolor=color1,markeredgewidth=1)
plt.errorbar(seps[:],adsorption[1,:], yerr = adsorption_err[1,:], marker='None', color='k', capsize=3, linestyle='None')
plt.plot(seps,adsorption[1,:],'s-',label='Cl$^-$',markersize=7, color=color2, markerfacecolor=matplotlib.colors.to_rgba(color2,0.7),markeredgecolor=color2,markeredgewidth=1)

plt.xlabel('Slit height ($\AA$)')
plt.ylabel('Fraction of adsorbed ions')
plt.grid('both',linestyle='--')
leg = plt.legend(frameon=True,framealpha=1)
leg.get_frame().set_edgecolor('k')
plt.savefig('frac_adsorbed.pdf', format='pdf', bbox_inches='tight')