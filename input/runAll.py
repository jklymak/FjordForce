import os
import subprocess


runModelName = 'runModelCedar.sh'

for todo in [f'Bute3d{runno}' for runno in range(51, 62)]:
    outstr = f"{todo} queued "
    res = subprocess.check_output(["sbatch", f"--job-name={todo}",
                        f"{runModelName}"])
    job = res.decode('utf8').split()[-1]
    outstr += f"{job} "
    res = subprocess.check_output(["sbatch", f"--job-name={todo}",
                        f"--dependency=afterok:{job}",
                        f"../python/runCrossSections.sh"])
    job = res.decode('utf8').split()[-1]
    outstr += f"{job} "
    # store info in a file
    print(outstr)
    with open(".joblog", "a") as joblog:
        joblog.write(outstr+"\n")
