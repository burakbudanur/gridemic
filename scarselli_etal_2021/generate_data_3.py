import os
import sys
import numpy as np
from subprocess import call

def replace_line(file_name, line_num, text):
    line_num = line_num - 1
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

    return

fname_submit = "submit.script"

for k in range(200):

    # Edit the submission script

    replace_line(fname_submit, 7,
        "#SBATCH --job-name=run3_"+str(k)+"\n")

    replace_line(fname_submit, 8,
        "#SBATCH --output=data/outrun3_"+str(k)+"\n")

    replace_line(fname_submit, 18,
        "srun --cpu_bind=verbose  python run3.py {}\n".format(k))

    # Submit the job
    call(['sbatch', fname_submit])
