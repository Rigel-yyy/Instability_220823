import argparse
import importlib

parser = argparse.ArgumentParser(description="Main entry for running programs on bjx slurm")
parser.add_argument('--id', type=int, help="slurm job array index", required=True)
parser.add_argument('--name', type=str, help="bjx main script stem name ", required=True)

sys_args = parser.parse_args()

bjx_script = importlib.import_module('bjx_main_src.' + sys_args.name)
bjx_script.run(job_array_id=sys_args.id)
