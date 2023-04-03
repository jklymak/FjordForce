# source ~/venvs/butewind/bin/activate
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xmitgcm
import sys, os
import glob

runname = sys.argv[1]

if len(sys.argv) > 2:
    snaptime = int(sys.argv[2]) * 3600
else:
    # get the last one:
    d = glob.glob(f'../results/{runname}/input/spinup.*.data')
    d.sort()
    parts = d[-1].split('.')
    print(parts)
    snaptime = int(parts[3])

duration = float(sys.argv[3])


try:
    os.mkdir(f'../results/{runname}/SnapsAll/')
except:
    pass

taumax = 0.225  # N/m^2
t  = [0, 1, duration+1, duration+2, 17]
tau = [0, taumax, taumax, 0, 0]


# get levels:
try:
    backgroundDen = xr.open_dataset('backgroundDen.nc')
except:
    with xmitgcm.open_mdsdataset(f'../results/{runname}/input/', endian='<',
                                iters=0) as ds0:
        ds = ds0.isel(time=0).sel(YC=1500, method="nearest").sel(XC=20000, method="nearest")
        backgroundDen = ds.THETA * -0.2 + ds.SALT * 0.75
        xr.Dataset( data_vars={'den': ('Z', backgroundDen.values)}, coords={'Z': ds.Z}).to_netcdf('backgroundDen.nc')
    backgroundDen = xr.open_dataset('backgroundDen.nc')

levelsZ = np.arange(0, -200, -25)-1.25
levels = np.interp(-levelsZ, -backgroundDen.Z.values, backgroundDen.den.values)
levdict = {}
for nn, lev in enumerate(levels):
    levdict[lev] = f'{levelsZ[nn]+1.25:1.0f}'

print(levels)

if True:
    for hours in range(24*17):
        snaptime = hours * 3600
        fig, axs = plt.subplot_mosaic([['stress', 'top'], ['cross', 'along'], ['Vcross', 'Ualong']],
                                constrained_layout=True, gridspec_kw={'width_ratios': [1, 3], 'height_ratios':[1, 2, 2]},
                                figsize=(9, 7))

        with xmitgcm.open_mdsdataset(f'../results/{runname}/input/', endian='<',
                                    iters=snaptime) as ds0:
            ds = ds0.isel(time=0)
            ds = ds.rename({'TRAC01':'O2'})
            ds['pden'] = ds.THETA * -0.2 + ds.SALT * 0.75
            ds['XC'] = ds.XC / 1e3
            ds['YC'] = ds.YC / 1e3
            ds['XG'] = ds.XG / 1e3
            ds['YG'] = ds.YG / 1e3

            ax = axs['stress']
            ax.plot(t, tau, linewidth=2)
            ax.axvline(snaptime/24/3600, lw=4, alpha=0.5)
            ax.set_ylabel('Stress $[N\,m^{-2}]$')
            ax.set_xlabel('Time $[d]$')
            ax.text(0.1, 0.1, f'Time {snaptime/24/3600:03.1f} d', fontfamily='monospace', transform=ax.transAxes)
            ax.tick_params(labelbottom=False, labeltop=True, bottom=False, top=True)


            ax = axs['along']
            dsal = ds.sel(YC=1.5, method="nearest").sel(XC=slice(0, 80))
            dsal['O2'] = dsal.O2.where(dsal.hFacC > 0.01)
            dsal['pden'] = dsal.pden.where(dsal.hFacC > 0.01)
            print(dsal)
            pc = ax.pcolormesh(dsal.XC, dsal.Z, dsal.O2,  vmin=0, vmax=300, cmap='turbo')
            cs = ax.contour(dsal.XC, dsal.Z, dsal.pden, levels=levels, linewidths=0.2, colors='0.4')
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.axvline(15, color='0.5', alpha=0.5, lw=3)
            fig.colorbar(pc, ax=ax, shrink=0.6)


            ax = axs['Ualong']
            ax.set_title('U $[m\,s^{-1}]$', fontsize='medium', loc='left')
            dsalg = ds.sel(YC=1.5, method="nearest").sel(XG=slice(0, 80))
            dsalg['UVEL'] = dsalg.UVEL.where(dsal.hFacW > 0.01)
            pc = ax.pcolormesh(dsalg.XG, dsalg.Z, dsalg.UVEL,  vmin=-0.5, vmax=0.5, cmap='RdBu_r')
            cs = ax.contour(dsal.XC, dsal.Z, dsal.pden, levels=levels, linewidths=0.2, colors='0.4')
            ax.tick_params(labelleft=False)
            ax.set_xlabel('X [km]')
            ax.axvline(15, color='0.5', alpha=0.5, lw=3)

            fig.colorbar(pc, ax=ax, shrink=0.6)

            ax = axs['cross']
            dsal = ds.sel(XC=15, method="nearest").sel(YC=slice(0, 3))
            dsal['O2'] = dsal.O2.where(dsal.hFacC > 0.01)
            dsal['pden'] = dsal.pden.where(dsal.hFacC > 0.01)
            pc = ax.pcolormesh(dsal.YC, dsal.Z, dsal.O2,  vmin=0, vmax=300, cmap='turbo',  )
            cs = ax.contour(dsal.YC, dsal.Z, dsal.pden, levels=levels, linewidths=0.2, colors='0.4')
            ax.clabel(cs, fmt=levdict, fontsize='small')
            ax.set_xlim(3, 0)
            ax.tick_params(labelbottom=False, labelleft=True)
            ax.axvline(1.5, color='0.5', alpha=0.5, lw=3)

            ax.set_ylabel('DEPTH [m]')

            fig.colorbar(pc, ax=ax, shrink=0.6)

            ax = axs['Vcross']
            ax.set_title('V $[m\,s^{-1}]$', fontsize='medium', loc='left')
            dsalg = ds.sel(XC=15, method="nearest").sel(YG=slice(0, 3))
            dsalg['VVEL'] = dsalg.VVEL.where(dsalg.hFacS > 0.01)
            pc = ax.pcolormesh(dsalg.YG, dsalg.Z, dsalg.VVEL,  vmin=-0.25, vmax=0.25, cmap='RdBu_r')
            cs = ax.contour(dsal.YC, dsal.Z, dsal.pden, levels=levels, linewidths=0.2, colors='0.4')
            ax.axvline(1.5, color='0.5', alpha=0.5, lw=3)
            ax.clabel(cs, fmt=levdict, fontsize='small')
            ax.set_xlim(3, 0)
            ax.set_xlabel('Y [km]')
            fig.colorbar(pc, ax=ax, shrink=0.6)


            ax = axs['top']
            dsal = ds.sel(Z=3.7, method="nearest").sel(XC=slice(0, 80)).sel(YC=slice(0, 3))
            dsal['O2'] = dsal.O2.where(dsal.hFacC > 0.01)
            dsal['pden'] = dsal.pden.where(dsal.hFacC > 0.01)
            pc = ax.pcolormesh(dsal.XC, dsal.YC, dsal.O2,  vmin=0, vmax=300, cmap='turbo',  )
            cs = ax.contour(dsal.XC, dsal.YC, dsal.pden, levels=levels, linewidths=0.2, colors='0.4')
            ax.clabel(cs, fmt=levdict, fontsize='small')
            ax.axvline(15, color='0.5', alpha=0.5, lw=3)
            ax.axhline(1.5, color='0.5', alpha=0.5, lw=3)

            fig.colorbar(pc, ax=ax, shrink=0.6, aspect=10)
            ax.tick_params(labelbottom=False, labelleft=True)
            ax.set_title('O2 $[\mu mol\ kg^{-3}]$', fontsize='medium', loc='left')

            for ax in axs:
                axs[ax].set_facecolor('0.4')
            axs['stress'].set_facecolor('1')
            fig.savefig(f'../results/{runname}/SnapsAll/frame{hours:04d}.png')
            plt.close(fig)
            del axs
            del fig

cmd = f'rm ../results/{runname}/SnapsAll/Movie3D.mp4'
os.system(cmd)

cmd = f'ffmpeg -r 20 -f image2 -start_number 1 -i ../results/{runname}/SnapsAll/frame%04d.png -vframes 1000 -vcodec libx264 -crf 20  -pix_fmt yuv420p ../results/{runname}/SnapsAll/Movie3D.mp4'
os.system(cmd)
cmd = f"ssh pender.seos.uvic.ca 'mkdir Dropbox/ButeWinds/{runname}'"
os.system(cmd)

cmd = f'rsync -a ../results/{runname}/SnapsAll/Movie3D.mp4 pender.seos.uvic.ca:Dropbox/ButeWinds/{runname}/Movie3D{runname}.mp4'
os.system(cmd)
