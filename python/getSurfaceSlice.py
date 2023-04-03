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
#snaptime = int(sys.argv[2])
#iternum = snaptime*3600
iternum = np.arange(13 * 24) * 3600

iternum = np.arange(24 * 16) * 3600
iternum = np.arange(12) * 24 * 3600

iternum = np.arange(11) * 24 * 3600

try:
    os.mkdir(f'../results/{runname}/slices/')
except:
    pass

with xmitgcm.open_mdsdataset(f'../results/{runname}/input/',
                             prefix=['spinup2d', 'spinup'],
                             endian='<', iters=iternum) as ds0:
    ds = ds0.isel(Z=1, Zl=1)
    ds.to_netcdf(f'../results/{runname}/slices/SurfaceSmall{runname}.nc')

os.system(f"ssh pender.seos.uvic.ca 'mkdir Dropbox/ButeWinds/{runname}/'")
os.system(f"rsync -av ../results/{runname}/slices/*.nc pender.seos.uvic.ca:Dropbox/ButeWinds/{runname}/ --exclude=SurfaceBig{runname}.nc")
