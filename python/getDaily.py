# source ~/venvs/butewind/bin/activate
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xmitgcm
import sys

runname = sys.argv[1]
# snaptime = int(sys.argv[2])*3600
taumax = 0.225  # N/m^2
t = np.arange(407*1.0)  # hours
taut = 0 * t
taut[t<=24] = np.arange(25) / 24 * taumax
taut[(t>24) & (t<(5*24))] = taumax
taut[(t>=5*24) & (t<6*24)] = np.arange(23, -1, -1) / 24 * taumax


with xmitgcm.open_mdsdataset(f'../results/{runname}/input/', endian='<',
                             iters=np.arange(17)*24*3600) as ds0:
    ds0 = ds0.isel(YC=0, YG=0)
    ds = ds0.rename({'TRAC01':'O2', 'TRAC02':'O2noEx'})
    ds['pden'] = ds.THETA * -0.2 + ds.SALT * 0.75
    ds.to_netcdf(f'../results/{runname}/input/Daily.nc')