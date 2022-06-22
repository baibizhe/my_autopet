import os
from easydict import EasyDict
import torch
C = EasyDict()
config = C
cfg = C

C.seed = 2022



""" Training setting """
# trainer
C.num_workers = 6
C.target_size= (128, 128, 128)
C.batch_size = 2

C.learning_rate = 1e-4
C.weight_decay = 0.98
C.dataPath= os.path.join("data","resize_128_data")
C.epochs = 100
C.lr_sheduler = "ExponentialLR"
C.optimizer_name = "adamw"
C.loss_name=""
C.aug_mode = "baseline"
C.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""" Model setting """
C.drop_out = True  # bayesian
C.drop_out_rate = 0.0

""" Wandb setting """
os.environ['WANDB_API_KEY'] = "55a895793519c48a6e64054c9b396629d3e41d10"
C.use_wandb = True
C.project_name = "autoPET"

""" Others """
C.save_ckpt = True

