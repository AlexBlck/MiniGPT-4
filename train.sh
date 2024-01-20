cd /mnt/fast/nobackup/users/aj00869/chapter4/minigpt/
source /opt/conda/bin/activate
conda init bash
conda activate /mnt/fast/nobackup/scratch4weeks/aj00869/conda_env/minigpt4
git checkout $1
/mnt/fast/nobackup/scratch4weeks/aj00869/conda_env/minigpt4/bin/torchrun --nproc-per-node 2 train.py --cfg-path train_configs/minigptv2_finetune.yaml


