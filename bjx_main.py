import argparse
import importlib
import time
from datetime import datetime

parser = argparse.ArgumentParser(description="Main entry for running programs on bjx slurm")
parser.add_argument('--id', type=int, help="slurm job array index", required=True)
parser.add_argument('--name', type=str, help="bjx main script stem name ", required=True)

sys_args = parser.parse_args()
time.sleep(10 * sys_args.id)
print(f'{datetime.now()}: Run with slurm job array id: {sys_args.id}')

bjx_script = importlib.import_module('bjx_main_src.' + sys_args.name)
bjx_script.run(job_array_id=sys_args.id)
