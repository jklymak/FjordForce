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
iternum = np.arange(24*3600, 16*24*3600, 24*3600)
iternum = (np.arange(11 * 24) + 23) *3600

iternum = np.arange(0, 24*24, 1) * 3600

Zind = 5


try:
    os.mkdir(f'../results/{runname}/slices/')
except:
    pass

newiter = []
notthere = []
for nn in iternum:
    if os.path.isfile(f'../results/{runname}/input/spinup.{nn:010d}.data'):
        newiter += [nn]
    else:
        notthere += [nn]


if notthere:
    print('Warning, not doing ', notthere)

if True:
    with xmitgcm.open_mdsdataset(f'../results/{runname}/input/',
                                 endian='<', iters=newiter,
                                 prefix=['spinup', 'spinup2d']) as ds0:
        ds = ds0.isel(Z=Zind, Zu=Zind, Zl=Zind, Zp1=Zind, YC=120, YG=120)
        ds.to_netcdf(f'../results/{runname}/slices/OneDepth{runname}Zind{Zind}.nc')

os.system(f"ssh pender.seos.uvic.ca 'mkdir Dropbox/ButeWinds/{runname}/'")
os.system(f"rsync -av ../results/{runname}/slices/*.nc pender.seos.uvic.ca:Dropbox/ButeWinds/{runname}/")
