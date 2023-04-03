import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xmitgcm
import sys, os

runname = sys.argv[1]
# snaptime = int(sys.argv[2])*3600
taumax = 0.225  # N/m^2
t = np.arange(407*1.0)  # hours


try:
    os.mkdir(f'../results/{runname}/Movie')
except:
    pass

if True:

    fig = plt.figure(constrained_layout=True, figsize=(7, 7))

    with xmitgcm.open_mdsdataset(f'../results/{runname}/input/', prefix=['spinup2d', 'spinup'], endian='<') as ds0:

        for ind in np.arange(len(ds0.time)):
            print(ind)
            ax = fig.subplots(3, 1, sharex=True, sharey=True)
            snaptime = ind * 3600
            ds = ds0.isel(time=ind)
            ds = ds.rename({'TRAC01':'O2'})
            ds['XC'] = ds.XC / 1e3
            ds['XG'] = ds.XG / 1e3
            ds = ds.isel(YC=120, YG=120)
            ds['UVEL'] = ds.UVEL.where(ds.THETA.values > 0)
            ds['O2'] = ds.O2.where(ds.THETA.values > 0)
            ds['THETA'] = ds.THETA.where(ds.THETA > 0)
            ds['pden'] = ds.THETA * -0.2 + ds.SALT * 0.75
            levels = np.arange(10, 22, 1/3)

            pc = ax[0].pcolormesh(ds.XC, ds.Z, ds.THETA, vmin=8.0, vmax=10, cmap='RdBu_r', rasterized=True)
            fig.colorbar(pc, ax=ax[0], shrink=0.8, extend='both')
            #ds.dT.plot.pcolormesh(ax=ax[0], vmin=-1, vmax=1, cmap='RdBu_r')
            ax[0].set_title(r'$\theta\ [^oC]$', loc='right', fontsize='medium')
                                                # levels=np.array([15, 20, 25, 26, 27, 28, 29, 29.25, 29.5, 29.75, 30, 30.05, 30.1, 30.2, 30.3]))

            pc = ax[1].pcolormesh(ds.XG, ds.Z, ds.UVEL, vmin=-0.25, vmax=0.25, cmap='RdBu_r', rasterized=True)
            fig.colorbar(pc, ax=ax[1], shrink=0.8, extend='both', )
            #ds.WVEL.plot.pcolormesh(ax=ax[1], vmin=0.03, vmax=-0.03, cmap='RdBu_r',  rasterized=True)
            ax[1].set_title(r'$U\ [m\,s^{-1}]$', loc='right', fontsize='medium')

            pc = ax[2].pcolormesh(ds.XC, ds.Z, ds.O2, vmin=0, vmax=320, cmap='turbo', rasterized=True)
            fig.colorbar(pc, ax=ax[2], shrink=0.8, extend='both', )
            ax[2].set_title(r'$O^2\ [\mu mol\,kg^{-1}]$', loc='right', fontsize='medium')

            for aa in ax:
                aa.contour(ds.XC, ds.Z, ds.pden, levels=levels, linewidths=0.7, colors='0.4')

            for i in range(3):
                ax[i].set_facecolor('0.6')

            ax[0].set_ylabel('DEPTH [m]')
            ax[2].set_xlabel('Distance from Head [km]')
            hours = ind
            ax[0].set_title(f'{hours:4d} [h]', loc='left')
            ax[0].set_xlim([-0.1, 140])
            ax[0].set_ylim([-210, 0])
            fig.savefig(f'../results/{runname}/Movie/frame{hours:04d}.png')
            #ax[0].set_xlim([-100, 25000])
            #fig.savefig(f'../results/{runname}/Movie/frame{hours:04d}Zoom.png')
            fig.clear()


cmd = f'ffmpeg -r 20 -f image2 -start_number 1 -y -i ../results/{runname}/Movie/frame%04d.png -vframes 1000 -vcodec libx264 -crf 12  -pix_fmt yuv420p ../results/{runname}/Movie/Full.mp4'
os.system(cmd)
cmd = f"ssh pender.seos.uvic.ca 'mkdir Dropbox/ButeWinds/{runname}'"
os.system(cmd)
cmd = f'rsync -a ../results/{runname}/Movie/Full.mp4 pender.seos.uvic.ca:Dropbox/ButeWinds/{runname}/Full.mp4'
os.system(cmd)
