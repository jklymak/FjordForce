import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from pylab import *
from shutil import copy
from os import mkdir
import shutil,os,glob
import scipy.signal as scisig
from maketopo import getTopo2D
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger(__name__)

runname='Bute06'
comments = 'Higher near-surface stratification; bigger domain to prevent seiching.  More horizontal diffusion; observed TS to 230 m; Qnet=500, uw=15 m/s, Non hydrostatic!!'



# to change U we need to edit external_forcing recompile

outdir0='../results/'+runname+'/'

indir =outdir0+'/indata/'

dx0=100.
dy0=100.


# model size
nx = 8 * 160
ny = 1
nz = 100

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

shutil.copy('../build/mitgcmuv', outdir+'/../build/mitgcmuv')
shutil.copy('../build/Makefile', outdir+'/../build/Makefile')
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
  shutil.copy('data.rbcs', outdir+'/data.rbcs')
except:
  pass

_log.info("Done copying files")

####### Make the grids #########

# Make grids:

##### Dx ######

dx = np.zeros(nx) + dx0
for i in range(nx-100, nx):
    dx[i] = dx[i-1] * 1.05


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
fig.savefig(outdir+'/figs/dx.pdf')

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
dz = np.ones(nz)
for i in range(1, nz):
    dz[i] = dz[i-1] * 1.0273

with open(indir+"/delZ.bin", "wb") as f:
	dz.tofile(f)
f.close()
z=np.cumsum(dz)
print(z)
####################
# temperature profile...
#
# temperature goes on the zc grid:
df = pd.read_csv('TempSal.csv')
T0 = 8 + np.ones(nz)
zs = np.array([0, 5, 20, 50, 100, 200, 500])
ts = np.array([7.4, 7.4, 7.4, 7.9, 8.0, 8.3, 8.06])
T0 = np.interp(z, zs, ts)

T0[z<220] = np.interp(z[z<220], df.Depth, df.Temperature)
T0[z>=220] = T0[z<220][-1]


with open(indir+"/TRef.bin", "wb") as f:
	T0.tofile(f)
#plot
plt.clf()
plt.plot(T0,z)
plt.savefig(outdir+'/figs/TO.pdf')

T0 = T0[:, np.newaxis] * (x[np.newaxis, :] * 0 + 1)
with open(indir+"/TInit.bin", "wb") as f:
	T0.tofile(f)
f.close()



#########################
# salinity profile...
#
# FRom
s = np.array([15, 15, 29.1, 29.6, 30.1, 30.6, 30.66])
S0 =  30.6 - 15*np.exp(-z / 20)

S0 = np.interp(z, zs, s)

S0[z<220] = np.interp(z[z<220], df.Depth, df.Salinity)
S0[z>=220] = S0[z>=220] - S0[z>=220][0] + S0[z<220][-1]
with open(indir+"/SRef.bin", "wb") as f:
	S0.tofile(f)
plt.clf()
plt.plot(S0, z)

plt.plot(s, zs, 'd' )
plt.savefig(outdir+'/figs/SO.pdf')

S0 = S0[:, np.newaxis] * (x[np.newaxis, :] * 0 + 1)
with open(indir+"/SInit.bin", "wb") as f:
	S0.tofile(f)

############################
# external wind stress
nt = 17 * 24
Cd = 1e-3
uw = 13  # m/s
taumax = Cd * uw**2  # N/m^2
t = np.arange(nt*1.0)  # hours
taut = 0 * t
taut[t<=24] = np.arange(25) / 24 * taumax
taut[(t>24) & (t<(6*24))] = taumax
taut[(t>=6*24) & (t<7*24)] = np.arange(23, -1, -1) / 24 * taumax
fig, ax = plt.subplots()
ax.plot(t, taut)
fig.savefig(outdir+'/figs/Tau.pdf')

taux = np.exp(-x/30000)
taux = 0.5-np.tanh((x-60e3)/30e3)/2

print(taux)
tau = taut[np.newaxis, :] * taux[:, np.newaxis]
print(np.shape(tau))
fig, ax = plt.subplots()
ax.pcolormesh(x, t,  tau.T, rasterized=True, vmin=-taumax, vmax=taumax, cmap='RdBu_r')
fig.savefig(outdir+'/figs/Tau.pdf')

with open(indir+'taux.bin', 'wb') as f:
    tau.T.tofile(f)

################################
# external heat flux
Qnetmax = 500
Qt = taut / taumax * Qnetmax

Q = Qt[np.newaxis, :] * taux[:, np.newaxis]
with open(indir+'Qnet.bin', 'wb') as f:
    Q.T.tofile(f)

###################################
# RBCS sponge
weight = np.zeros((nz, ny, nx))
weight[:, -100:] = np.arange(0, 1, 100)**1.5
with open(indir+'spongeweight.bin', 'wb') as f:
    weight.tofile(f)

# force to zero velocity to prevent reflections.

weight = np.zeros((nz, ny, nx))
with open(indir+'Uforce.bin', 'wb') as f:
    weight.tofile(f)


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
