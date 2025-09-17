#!/bin/bash
#SBATCH --job-name=run_nzgmdb
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --exclude=n04
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
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
python $SCRATCH/code/nzgmdb/nzgmdb/scripts/run_nzgmdb.py run-full-nzgmdb /scratch/jobs/jri83/runs/test_run 2000-01-01 2024-12-31 /scratch/jobs/jri83/code/nzcvm_data /scratch/jobs/jri83/code/gm_classifier /home/jri83/miniforge3/etc/profile.d/conda.sh "conda activate gmc" "conda activate gmc_predict" --only-record-ids-ffp /scratch/jobs/jri83/runs/test_run/bypass_datetime_waves_sep_12.csv /scratch/projects/rch-quakecore/ko_matrices --n-procs 24 --gmc-procs 6 --checkpoint