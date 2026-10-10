#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1   # typed: a generic gpu:1 is routed to the Blackwell pool, which this container (CUDA 12.6) cannot run on; nvidia_h200 also works
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=smoke_%j.log
set -e
module load apptainer
cd $WORK
apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER dmasif_sandbox python -c "
import numpy as np, torch, pykeops, torch_geometric
print('numpy', np.__version__, '| torch', torch.__version__, '| pyg', torch_geometric.__version__)
print(torch.cuda.get_device_name())
x = torch.from_numpy(np.arange(6, dtype=np.float32)).cuda(); print('numpy<->torch ok:', x.cpu().numpy().sum())
pykeops.test_torch_bindings()
"
echo SMOKE_TEST_PASSED
