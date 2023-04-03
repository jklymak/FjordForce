#!/bin/bash
#SBATCH --account=def-jklymak
#SBATCH --mail-user=jklymak@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:15
#SBATCH --mem=64G

cd ${SLURM_SUBMIT_DIR}

top=${SLURM_JOB_NAME//"-monitor"/}
echo ${top}

mkdir ../results/${top}/Snaps
# mkdir ../results/${top}/SnapsTop

source /home/jklymak/venvs/butewind/bin/activate

runagain=false
if python plot_snap.py ${top}; then
    runagain=true
fi
if ${runagain}; then
    #python plot_snapTop.py ${top}

    chmod a+rx ../results/${top}/Snaps
    chmod a+rx ../results/${top}/Snaps/*
    #chmod a+rx ../results/${top}/SnapsTop
    #chmod a+rx ../results/${top}/SnapsTop/*
    rsync -av ../results/${top}/Snaps pender.seos.uvic.ca:Sites/ButeWinds/${top}/
    #rsync -av ../results/${top}/SnapsTop pender.seos.uvic.ca:Sites/ButeWinds/${top}/

    ssh pender.seos.uvic.ca "chmod a+rx Sites/ButeWinds/${top}/Snaps;chmod a+rx Sites/ButeWinds/${SLURM_JOB_NAME}/Snaps/*.png"
    #ssh pender.seos.uvic.ca "chmod a+rx Sites/ButeWinds/${top}/SnapsTop;chmod a+rx Sites/ButeWinds/${SLURM_JOB_NAME}/SnapsTop/*.png"

    sbatch --job-name=${SLURM_JOB_NAME} --exclude=cdr2098  --begin=now+2hour runMonitor.sh
fi

