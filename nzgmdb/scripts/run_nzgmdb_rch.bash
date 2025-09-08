#!/bin/bash
#SBATCH --job-name=run_nzgmdb
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:30:00
#SBATCH --exclude=n12
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
source $HOME/.bashrc
mamba activate nzgmdb
echo ===== ENVIRONMENT =====
echo PYTHON  $(which python)
echo VERSION $(python --version)
echo PWD     $(pwd)
echo SCRATCH $SCRATCH
module list
echo =======================
mkdir $SCRATCH/runs/test_run
python $SCRATCH/code/nzgmdb/nzgmdb/scripts/run_nzgmdb.py generate-site-table-basin $SCRATCH/runs/test_run