eval "$(/home/djyjyh/miniconda3/bin/conda shell.bash hook)"
conda activate rob_env
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
