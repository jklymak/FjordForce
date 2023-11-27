# on narval `source ~/venvs/bute_venv_narval/bin/activate`
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from pylab import *
from shutil import copy
from os import mkdir
import shutil,os,glob
#import scipy.signal as scisig
from maketopo import getTopo2D
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger(__name__)

runname='Fjord4'
comments = """Lower res (dx=100, nz=20) Atmopsheric Load; RBCS Compiled forcing"""

outdir0='../results/'+runname+'/'

indir =outdir0+'/indata/'

dx0=100.
dy0=100.


# model size
nx = 6 * 200
ny = 1
nz = 20

_log.info('nx %d ny %d', nx, ny)

def lininc(n,Dx,dx0):
    a=(Dx-n*dx0)*2./n/(n+1)
    dx = dx0+arange(1.,n+1.,1.)*a
    return dx


#### Set up the output directory
backupmodel=1
if backupmodel:
  try:
    mkdir(outdir0)
  except:
    import datetime
    import time
    ts = time.time()
    st=datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    shutil.move(outdir0[:-1],outdir0[:-1]+'.bak'+st)
    mkdir(outdir0)

    _log.info(outdir0+' Exists')

  outdir=outdir0
  try:
    mkdir(outdir)
  except:
    _log.info(outdir+' Exists')
  outdir=outdir+'input/'
  try:
    mkdir(outdir)
  except:
    _log.info(outdir+' Exists')
  try:
      mkdir(outdir+'/figs/')
  except:
    pass

  copy('gendata.py',outdir)
else:
  outdir=outdir+'input/'

## Copy some other files
_log.info( "Copying files")

try:
  shutil.rmtree(outdir+'/../code/')
except:
  _log.info("code is not there anyhow")
shutil.copytree('../code', outdir+'/../code/')
shutil.copytree('../python', outdir+'/../python/')

try:
  shutil.rmtree(outdir+'/../build/')
except:
  _log.info("build is not there anyhow")
_log.info(outdir+'/../build/')
mkdir(outdir+'/../build/')

# copy any data that is in the local indata
shutil.copytree('../indata/', outdir+'/../indata/')
try:
  shutil.copy('../build/mitgcmuv', outdir+'/../build/mitgcmuv')
  shutil.copy('../build/Makefile', outdir+'/../build/Makefile')
except:
  pass
shutil.copy('data', outdir+'/data')
shutil.copy('eedata', outdir)
shutil.copy('data.kl10', outdir)
try:
  shutil.copy('data.kpp', outdir)
except:
  pass

try:
    shutil.copy('data.obcs', outdir)
except:
    pass
try:
  shutil.copy('data.diagnostics', outdir)
except:
  pass
try:
  shutil.copy('data.pkg', outdir+'/data.pkg')
except:
  pass
try:
  shutil.copy('data.ptracers', outdir+'/data.ptracers')
except:
  pass
try:
  shutil.copy('data.rbcs', outdir+'/data.rbcs')
except:
  pass

_log.info("Done copying files")

####### Make the grids #########

# Make grids:

##### Dx ######

dx = np.zeros(nx) + dx0
for i in range(nx-200, nx):
    dx[i] = dx[i-1] * 1.025


# dx = zeros(nx)+100.
x=np.cumsum(dx)
x=x-x[0]
maxx=np.max(x)
_log.info('XCoffset=%1.4f'%x[0])

##### Dy ######

dy = np.array(dx0)
y=np.cumsum(dy)

# save dx and dy
with open(indir+"/delX.bin", "wb") as f:
  dx.tofile(f)
f.close()
with open(indir+"/delY.bin", "wb") as f:
  dy.tofile(f)
f.close()

# some plots
fig, ax = plt.subplots(2,1)
ax[0].plot(x/1000.,dx)
ax[1].plot(y/1000.,dy)
#xlim([-50,50])
fig.savefig(outdir+'/figs/dx.png')

######## Bathy ############
# get the topo:
xd = np.array([0, 3, 23, 80, 83, 88, 91, 111, 140, 150, 10000])
dd = np.array([0, 300, 500, 500, 400, 400, 500, 300, 300, 500, 500])
d = np.zeros((ny,nx))
d[0, :] = np.interp(x, xd*1000, -dd)
with open(indir+"/topog.bin", "wb") as f:
  d.tofile(f)
f.close()

_log.info(np.shape(d))

fig, ax = plt.subplots(2,1)
_log.info('%s %s', np.shape(x), np.shape(d))
print(y)
ax[0].plot(x/1.e3,d[0,:].T)
pcm=ax[1].pcolormesh(x/1.e3,y/1.e3,d,rasterized=True)
fig.colorbar(pcm,ax=ax[1])
fig.savefig(outdir+'/figs/topo.png')

##################
# dz:
# dz is from the surface down (right?).  Its saved as positive.
dz = np.ones(nz) * 25
for i in range(15, nz):
    dz[i] = dz[i-1] * 1.03

with open(indir+"/delZ.bin", "wb") as f:
	dz.tofile(f)
f.close()
z=np.cumsum(dz)
print(z)
fig, ax = plt.subplots()
ax.plot(z, dz)
fig.savefig(outdir+'/figs/dz.png')

####################
# temperature profile...
#
# temperature goes on the zc grid:
df = pd.read_csv('TempSal.csv')
T0 = 8 + np.ones(nz)
zs = np.array([0, 5, 20, 50, 100, 200, 500, 600])
ts = np.array([7.4, 7.4, 7.4, 7.9, 8.0, 8.3, 8.06, 8.06])
T0 = np.interp(z, zs, ts)

T0[z<220] = np.interp(z[z<220], df.Depth, df.Temperature)
T0[z>=220] = T0[z<220][-1]


with open(indir+"/TRef.bin", "wb") as f:
	T0.tofile(f)
#plot
plt.clf()
plt.plot(T0,z)
plt.savefig(outdir+'/figs/TO.png')

T0 = T0[:, np.newaxis] * (x[np.newaxis, :] * 0 + 1)
with open(indir+"/TInit.bin", "wb") as f:
	T0.tofile(f)
f.close()



#########################
# salinity profile...
#
# FRom
s = np.array([15, 15, 29.1, 29.6, 30.1, 30.6, 30.66, 30.66])
S0 =  30.6 - 15*np.exp(-z / 20)

S0 = np.interp(z, zs, s)

S0[z<220] = np.interp(z[z<220], df.Depth, df.Salinity)
S0[z>=220] = S0[z>=220] - S0[z>=220][0] + S0[z<220][-1]
with open(indir+"/SRef.bin", "wb") as f:
	S0.tofile(f)
plt.clf()
plt.plot(S0, z)

plt.plot(s, zs, 'd' )
plt.savefig(outdir+'/figs/SO.png')

S0 = S0[:, np.newaxis] * (x[np.newaxis, :] * 0 + 1)
with open(indir+"/SInit.bin", "wb") as f:
	S0.tofile(f)

## Make timeseries
nt = 17 * 24
t = np.arange(nt*1.0)  # hours

if False:

  ############################
  # external wind stress
  Cd = 1e-3
  uw = 15  # m/s
  taumax = Cd * uw**2  # N/m^2
  taut = 0 * t
  taut[t<=24] = np.arange(25) / 24 * taumax
  taut[(t>24) & (t<(6*24))] = taumax
  taut[(t>=6*24) & (t<7*24)] = np.arange(23, -1, -1) / 24 * taumax

  taux = np.exp(-x/30000)
  taux = 0.5-np.tanh((x-60e3)/30e3)/2

  print(taux)
  tau = taut[np.newaxis, :] * taux[:, np.newaxis]
  print(np.shape(tau))
  fig, ax = plt.subplots()
  ax.pcolormesh(x, t,  tau.T, rasterized=True, vmin=-taumax, vmax=taumax, cmap='RdBu_r')
  fig.savefig(outdir+'/figs/Tau.png')

  with open(indir+'taux.bin', 'wb') as f:
      tau.T.tofile(f)

  ################################
  # external heat flux
  Qnetmax = 500
  Qt = taut / taumax * Qnetmax

  Q = Qt[np.newaxis, :] * taux[:, np.newaxis]
  with open(indir+'Qnet.bin', 'wb') as f:
      Q.T.tofile(f)


amp = 1   # m
om = np.pi * 2 / 12.4  # rad/h

tidet = amp * np.sin( om * t ) * 9.81 * 1000  # kg m^-1 s^-2

tidex = np.zeros(nx)
tidex[x>=120e3] = 1.0  # I don't think there is any reason to be gentle for this forcing...

tide = tidet[np.newaxis, :] * tidex[:, np.newaxis]
with open(indir+'atmosphere.bin', 'wb') as f:
    tide.T.tofile(f)


###################################
# RBCS sponges
# We are doing this every 1860 s for 44640 s

t = np.arange(0, 44640, 1860)
nt = len(t)
weight = np.zeros((nz, ny, nx))
weight[..., -100:] = np.linspace(0, 1, 100)**1.5
# print(weight)
with open(indir+'spongeweight.bin', 'wb') as f:
    weight.tofile(f)
fig, ax = plt.subplots()

ax.plot(x, weight[0, 0, :])


weight = np.zeros((ny, nx))
weight[..., -130:] = np.linspace(0, 1, 130)**1.5
with open(indir+'etaweight.bin', 'wb') as f:
    weight.tofile(f)

ax.plot(x, weight[0, :])
fig.savefig(outdir+'/figs/weight.png')


# force to zero velocity to prevent reflections.
# note that we also force T and S to Tinit and Sinit
weight = np.zeros((nt, nz, ny, nx))
with open(indir+'Uforce.bin', 'wb') as f:
    weight.tofile(f)
weight = np.zeros((nt, nz, ny, nx)) + S0[np.newaxis, :, np.newaxis, :]
with open(indir+'Sforce.bin', 'wb') as f:
    weight.tofile(f)
weight = np.zeros((nt, nz, ny, nx)) + T0[np.newaxis, :, np.newaxis, :]
with open(indir+'Tforce.bin', 'wb') as f:
    weight.tofile(f)
# seasurface height forcing.  tidal!

amp = 1  # m
om = np.pi * 2 / 12.4 / 3600 # rad/s

weight = np.zeros((nt, ny, nx)) + (amp*np.sin(om * t))[:, np.newaxis, np.newaxis]
with open(indir+'Etaforce.bin', 'wb') as f:
    weight.tofile(f)

print(weight)

#### Initial O2

O2 = np.zeros((nz, ny, nx))
O2z = np.interp(z, [0, 25, 120, 150, 1000], [7, 5, 2, 2.7, 2.7]) / 22.4 * 1e3  # umol/kg
O2 = O2 + O2z[:, np.newaxis, np.newaxis]
with open(indir+'O2.bin', 'wb') as f:
    O2.tofile(f)
with open(indir+'O2n.bin', 'wb') as f:
    O2.tofile(f)

fig, ax = plt.subplots()
ax.plot(O2z, z)
fig.savefig(outdir+'/figs/O2.png')


_log.info('All Done!')

_log.info('Archiving to home directory')

try:
    shutil.rmtree('../archive/'+runname)
except:
    pass

shutil.copytree(outdir0+'/input/', '../archive/'+runname+'/input')
shutil.copytree(outdir0+'/python/', '../archive/'+runname+'/python')
shutil.copytree(outdir0+'/code', '../archive/'+runname+'/code')

_log.info('doing this via git!!')

os.system(f'git commit -a -m "gendata for {runname}: {comments}"')
os.system('git push origin main')
os.system(f'git checkout -B {runname}')
os.system(f'git push origin {runname}')
os.system('git checkout main')


exit()
