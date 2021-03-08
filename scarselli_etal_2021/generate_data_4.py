import os
import sys
import numpy as np
from scipy.optimize import root_scalar
from subprocess import call

def replace_line(file_name, line_num, text):
    line_num = line_num - 1
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

    return

eps = np.linspace(0,1,101)

fname_submit = "submit.script"

for k in range(len(eps)):

    # Edit the submission script

    replace_line(fname_submit, 7,
        "#SBATCH --job-name=run4_"+str(k)+"\n")

    replace_line(fname_submit, 8,
        "#SBATCH --output=data/outrun4_"+str(k)+"\n")

    replace_line(fname_submit, 18,
        "srun --cpu_bind=verbose  python run4.py {:04.2f}\n".format(eps[k]))

    # Submit the job
    call(['sbatch', fname_submit])