#!/bin/bash
#SBATCH --account=def-jklymak
#SBATCH --mail-user=jklymak@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:15
#SBATCH --mem=64G

cd ${SLURM_SUBMIT_DIR}


printf "workdir? ${SLURM_SUBMIT_DIR}\n"
pwd

top=${SLURM_JOB_NAME//"-plotAll"/}
echo ${top}

source /home/jklymak/venvs/butewind/bin/activate

printf "done activate"
echo `which python`
python plotDataSnap.py ${top} 168 5

