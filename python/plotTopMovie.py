# source ~/venvs/butewind/bin/activate
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
taut = 0 * t
taut[t<=24] = np.arange(25) / 24 * taumax
taut[(t>24) & (t<(5*24))] = taumax
taut[(t>=5*24) & (t<6*24)] = np.arange(23, -1, -1) / 24 * taumax

try:
    os.mkdir(f'../results/{runname}/topMovie/')
except:
    print('Coldnt make directory, probably already there')
    pass
print('Done making')

if False:

    with xmitgcm.open_mdsdataset(f'../results/{runname}/input/', endian='<', iters=np.arange(12*24, dtype=int)*3600) as ds0:
        for time in range(0, 12*24):
            print(f'time {time}')
            fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True, figsize=(7, 7))
            ds = ds0.isel(time=time, Z=0)
            ds = ds.rename({'TRAC01':'O2'})
            ds['pden'] = ds.THETA * -0.2 + ds.SALT * 0.75
            levels = np.arange(10, 22, 1/3)
            ds.THETA.plot.pcolormesh(ax=ax[0], vmin=8.0, vmax=10, cmap='RdBu_r',  rasterized=True)
            #ds.dT.plot.pcolormesh(ax=ax[0], vmin=-1, vmax=1, cmap='RdBu_r')

            ds.pden.plot.contour(ax=ax[0], levels=levels, linewidths=0.2, colors='0.4')
                                                # levels=np.array([15, 20, 25, 26, 27, 28, 29, 29.25, 29.5, 29.75, 30, 30.05, 30.1, 30.2, 30.3]))

            ds.UVEL.plot.pcolormesh(ax=ax[1], vmax=0.6, vmin=-0.6, cmap='RdBu_r',  rasterized=True)

            ds.pden.plot.contour(ax=ax[1], levels=levels,linewidths=0.2, colors='0.4')
                                                # levels=np.array([15, 20, 25, 26, 27, 28, 29, 29.25, 29.5, 29.75, 30, 30.05, 30.1, 30.2, 30.3]))

            ds.ETAN.plot.pcolormesh(ax=ax[2], vmin=-0.075, vmax=0.075, cmap='RdBu_r', rasterized=True)
            #(ds.O2-ds.O2noEx).plot.pcolormesh(ax=ax[2], vmin=-100, vmax=100, cmap='RdBu_r')
            ds.pden.plot.contour(ax=ax[2], levels=levels,linewidths=0.2, colors='0.4')
                                                # levels=np.array([15, 20, 25, 26, 27, 28, 29, 29.25, 29.5, 29.75, 30, 30.05, 30.1, 30.2, 30.3]))

            for ii in range(3):
                ax[ii].set_title('')
            hours = time
            ax[0].set_title(f'{hours:4d} [h]: $\\tau = {taut[hours]:2.2f}$', loc='left')
            ax[0].set_xlim([-100, 240000])
            ax[0].set_ylim([0, 4e3])
            fig.savefig(f'../results/{runname}/topMovie/Topframe{hours:04d}.png')
            plt.close(fig)
            print(f'Done time {time}')

cmd = f'ffmpeg -r 20 -f image2 -start_number 1 -i ../results/{runname}/topMovie/Topframe%04d.png -vframes 1000 -vcodec libx264 -crf 25  -pix_fmt yuv420p ../results/{runname}/topMovie/topMovie.mp4'
os.system(cmd)

    #ax[0].set_xlim([-100, 25000])
    #fig.savefig(f'doc/Crossframe{hours:04d}Zoom.png')
