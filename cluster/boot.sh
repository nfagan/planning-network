module purge
module load anaconda3/2024.02
conda env create -f environment.yml -p /scratch/naf264/conda/pn0
conda activate /scratch/naf264/conda/pn0