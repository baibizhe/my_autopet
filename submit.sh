#!/bin/bash
#SBATCH --mail-user=bbzatyddounan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output/%N-%j.out
#SBATCH --gres=gpu:3

#SBATCH --cpus-per-task=4 # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=160000M       #FALRE22: 200GB 250LABEL是可以的 Memory proportional to GPUs: 32000 Cedar, 64000 Graham.     
#SBATCH --time=23:29:00
module load python/3.8
source /home/baibizhe/scratch/aienv/bin/activate 
# python check_file.py
# python main.py --roi_x 128 --roi_y 128 --roi_z 128 --sw_batch_size 1  
python -m torch.distributed.launch --nproc_per_node=3 --master_port=11223 main.py --batch_size=3 --num_steps=450000 --lrdecay --eval_num=500 --lr=4e-4 --decay=1e-5 --roi_x 96 --roi_y 96 --roi_z 96  




