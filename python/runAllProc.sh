
runname=$1
jobid=$2

echo ${runname}

sbatch --job-name=${runname} --dependency=afterok:${jobid} runGetSurface.sh
sbatch --job-name=${runname} --dependency=afterok:${jobid} runNsSections.sh
sbatch --job-name=${runname} --dependency=afterok:${jobid} runCrossSections.sh
sbatch --job-name=${runname} --dependency=afterok:${jobid} runGetDens.sh
