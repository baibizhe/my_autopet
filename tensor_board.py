import wandb
import torch
import torchvision
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
try:
    from skimage.util.montage import montage2d
except ImportError as e:
    # print('scikit-image is too new, ',e)
    from skimage.util import montage as montage2d


class Tensorboard:
    def __init__(self, config):
        os.system("wandb login")
        os.system("wandb {}".format("online" if config.use_wandb else "offline"))

        self.tensor_board = wandb.init(project=config.project_name,
                                       name=config.run_name,
                                       config=config)
        self.ckpt_root = 'saved'
        self.ckpt_path = os.path.join(self.ckpt_root, config.run_name)
        self.visual_root_path = os.path.join(self.ckpt_path, 'history_images')
        self.visual_results_root = os.path.join(self.visual_root_path, 'results')
        self._safe_mkdir(self.ckpt_root)
        self._safe_mkdir(self.ckpt_path)
        self._safe_mkdir(self.visual_root_path)
        self._safe_mkdir(self.visual_results_root)
        self._safe_mkdir(self.ckpt_root, config.run_name)

    def upload_wandb_info(self, info_dict, current_step=0):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info]})
        return



    def produce_2d_slice(self, image, label, pred):
        image = image[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = torchvision.utils.make_grid(image, 5, normalize=True)
        wandb.log({"volume_slice": [wandb.Image(grid_image,
                                                caption="x")]})

        outputs_soft = torch.softmax(pred, 1)
        image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = torchvision.utils.make_grid(image, 5, normalize=False)
        wandb.log({"output_slice": [wandb.Image(grid_image,
                                                caption="y_tilde")]})

        image = label[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1).float()
        grid_image = torchvision.utils.make_grid(image, 5, normalize=False)
        wandb.log({"label_slice": [wandb.Image(grid_image,
                                               caption="y")]})
        return



    @staticmethod
    def _safe_mkdir(parent_path, build_path=None):
        if build_path is None:
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)
        else:
            if not os.path.exists(os.path.join(parent_path, build_path)):
                os.mkdir(os.path.join(parent_path, build_path))
        return

    def save_ckpt(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, name))
        return

    def upload_wandb_image(self, train_predict, train_label):
        label_image = torch.argmax(train_label[0], 0)
        label_image_middle = label_image[int(len(label_image) * 0.4):int(len(label_image) * 0.6)]
        square_img_label_image = montage2d(label_image_middle.cpu().detach().numpy())


        predict_image = torch.argmax(train_predict[0], 0)
        predict_image_middle = predict_image[int(len(predict_image) * 0.4):int(len(predict_image) * 0.6)]
        square_img_predict_image = montage2d(predict_image_middle.cpu().detach().numpy())
        self.tensor_board.log({"label_image": wandb.Image((square_img_label_image*255),caption="label_image")})
        self.tensor_board.log({"predict_image": wandb.Image((square_img_predict_image*255),caption="predict_image")})

