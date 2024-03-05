cd /mnt/fast/nobackup/users/aj00869/chapter4/minigpt/
source /opt/conda/bin/activate
conda init bash
conda activate /mnt/fast/nobackup/scratch4weeks/aj00869/conda_env/mgp
git checkout $1
/mnt/fast/nobackup/scratch4weeks/aj00869/conda_env/mgp/bin/torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigptv2_finetune.yaml


