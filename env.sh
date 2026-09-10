CONDA_BIN="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate rob_env
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
