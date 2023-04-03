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
from replace_data import replace_data

backupmodel = True
logging.basicConfig(level=logging.INFO)

_log = logging.getLogger(__name__)

duration = 12
initial = 0
startday = 7
wind = 20  # m/s *3.6 to get km/h, wind**2 / 1000 to get stress
uw = wind
lat = 45
f0 = 1e-4 * np.sin(lat * np.pi / 180) / np.sin(45 * np.pi / 180)
wavey = False
Nsq0 = 3.44e-4
tAlpha = 0.0e-4
sBeta = 7.4e-4
Nsqfac = 1
Nsq0 = Nsq0 * Nsqfac

basename='Bute3d43'
runname = f'{basename}Stop{startday}d'
comments = f"""
Pickup from {basename}, stopped after {startday} days.
"""

origdir0 = '../results/' + basename + '/'
outdir0 = '../results/' + runname + '/'

indir = outdir0 + '/indata/'

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

tocp = ['/indata/', '/build/', '/code/']

for tc in tocp:
    shutil.copytree(origdir0 + tc,
                    outdir0 + tc)

mkdir(outdir0 + 'python/')

for file in glob.glob(origdir0 + 'python/*.sh'):
    shutil.copy(file, outdir0 + 'python/')
for file in glob.glob(origdir0 + 'python/*.py'):
    shutil.copy(file, outdir0 + 'python/')

mkdir(outdir0 + 'input/')

for file in glob.glob(origdir0 + 'input/data*'):
    shutil.copy(file, outdir0 + 'input/')
for file in glob.glob('./*.py'):
    shutil.copy(file, outdir0 + 'input/')
for file in glob.glob('./*.sh'):
    shutil.copy(file, outdir0 + 'input/')
shutil.copy(origdir0 + 'input/eedata',
            outdir0 + 'input/eedata')

start = 24 * 3600 * startday

replace_data(outdir0 + 'input/data', 'startTime', f'{start}')
replace_data(outdir0 + 'input/data', 'endTime', f'{start+duration*24*3600}')
replace_data(outdir0 + 'input/data', 'zonalWindFile', '')

shutil.copy(origdir0 + f'input/pickup.{start:010d}.data',
            outdir0 + f'input/pickup.{start:010d}.data')
shutil.copy(origdir0 + f'input/pickup.{start:010d}.meta',
            outdir0 + f'input/pickup.{start:010d}.meta')
shutil.copy(origdir0 + f'input/pickup_ptracers.{start:010d}.data',
            outdir0 + f'input/pickup_ptracers.{start:010d}.data')
shutil.copy(origdir0 + f'input/pickup_ptracers.{start:010d}.meta',
            outdir0 + f'input/pickup_ptracers.{start:010d}.meta')


try:
    shutil.rmtree('../archive/' + runname)
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
