import sys
import os
import subprocess

# os.system('sq')
#out=subprocess.check_output('squeue -h -t RUNNING -u jklymak --Format="Name:70"',
#                            shell=True, text=True)
#out = out.splitlines()
# print(out)
#for dd in out:
print(sys.argv)
dd = sys.argv[1]

os.system(f"grep -E 'advcfl_wvel_max|advcfl_uvel_max|time_seconds|dynstat_uvel_max' ../results/{dd}/input/STDOUT.0000 | tail -n 9")
ot = subprocess.check_output(f"grep -E 'time_seconds' ../results/{dd}/input/STDOUT.0000 | tail -n 1",
        shell=True, text=True)

st = ot.split(' ')[-1]
print(f'{float(st)} seconds')

days = float(st)/3600
print(f'{float(days)} hours')

days = float(st)/24/3600
print(f'{days:6.2f} days')