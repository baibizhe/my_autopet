import os

import monai
import torch
from timm.utils import AverageMeter
from TrainConfig import config

from get_class_data_loaders import get_class_data_loaders
from get_training_components import get_wandb


def classification(config):
    if config.use_wandb:
        mywandb = get_wandb(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.Densenet(spatial_dims=3, in_channels=2, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), 1e-2, nesterov=True,momentum=0.99)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-5)

    val_interval = 2
    best_metric = -1
    train_loader, val_loader = get_class_data_loaders(None)
    sigmoid_l = torch.nn.Sigmoid()
    for epoch in range(500):
        train_info ={}
        loss_one_epoch = train_one_epoch(device,  loss_function, model, optimizer, sigmoid_l, train_loader)

        if epoch % val_interval == 0:
            valid_metric_epoch = valid_one_epoch(device,  model, sigmoid_l, val_loader)
            if valid_metric_epoch > best_metric:
                if not os.path.exists("saved_models"):
                    os.mkdir("saved_models")
                best_metric = valid_metric_epoch
                torch.save(model.state_dict(), os.path.join("saved_models", "classification_model"))
                print("saved new best metric model")
        train_info["classification train loss"] = loss_one_epoch
        train_info["classification valid acc"] = valid_metric_epoch
        if config.use_wandb:
            mywandb.upload_wandb_info(train_info)
    print(train_info)



def valid_one_epoch( device,  model, sigmoid_l, val_loader):
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].long().squeeze(1).to(
                device)
            val_outputs = sigmoid_l(model(val_images))
            # print(val_outputs,val_outputs.argmax(dim=1),val_labels)
            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count


    return  metric


def train_one_epoch(device,  loss_function, model, optimizer, sigmoid_l, train_loader):
    model.train()
    losses = AverageMeter()

    for batch_data in train_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].long().to(device)
        optimizer.zero_grad()
        outputs = sigmoid_l(model(inputs))
        loss = loss_function(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        losses.update(loss.item(),inputs.shape[0])
    return losses.avg

if __name__ == "__main__":
    classification(config)
