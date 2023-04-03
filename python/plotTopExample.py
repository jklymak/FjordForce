# source ~/venvs/butewind/bin/activate
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xmitgcm
import sys, os

"""
Plot plan view of the numerical setup, largescale
"""

runname = sys.argv[1]
snaptime = int(sys.argv[2])
iternum = snaptime*3600

etalevels = np.arange(-1, 1, 0.1)
with xmitgcm.open_mdsdataset(f'../results/{runname}/input/', endian='<', iters=iternum) as ds0:
    ds = ds0.isel(Z=1, time=0)
    print(ds.coords)
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 4))
    ds['UVEL'] = ds.UVEL.where(ds.hFacW > 0)
    ds['ETAN'] = ds.ETAN.where(ds.hFacC > 0)
    ax = axs[0]
    ax.pcolormesh(ds.XC / 1e3, ds.YC / 1e3, ds.UVEL, vmax=0.6, vmin=-0.6, cmap='RdBu_r',  rasterized=True)
    ax.contour(ds.XC / 1e3, ds.YC / 1e3, ds.ETAN, vmin=-0.075, vmax=0.075, levels=etalevels, colors='k', rasterized=True)
    ax.set_aspect(1)
    ax.set_facecolor('0.5')
    ax.set_title('a) Receiving domain', loc='left', fontsize='medium')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')

    ax = axs[1]
    ax.pcolormesh(ds.XC / 1e3, ds.YC / 1e3, ds.UVEL, vmax=0.6, vmin=-0.6, cmap='RdBu_r',  rasterized=True)
    ax.contour(ds.XC / 1e3, ds.YC / 1e3, ds.ETAN, vmin=-0.075, vmax=0.075, levels=etalevels, colors='k', rasterized=True)
    ax.set_facecolor('0.5')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 3)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_title('b) Active domain', loc='left', fontsize='medium')
    inset = ax.inset_axes([0, 1.05, 1, 0.3])
    inset.plot(ds.XC / 1e3, 0.5-np.tanh((ds.XC-60e3)/30e3)/2)
    inset.set_xlim(0, 200)
    inset.set_ylim(0.001, 1.1)
    inset.set_xticks(inset.get_xticks(), labels=[])
    inset.set_ylabel(r'$\frac{\tau}{\tau_{max}}$')
    fig.savefig(f'./figs/geo{runname}{snaptime}.png')

