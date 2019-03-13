
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms 
import tensorboardX

from data import CheXpertDataset
import config
import models
import experiment


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def train_func(dataloader, model, optimizer, loss_criterion, device, summary_writer):
    model.to(device)
    model.train()

    global_iter = 0
    for epoch in range(config.epochs):
        iter_ = 0 
        for data, labels, masks in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward
            output = model(data)

            loss = loss_criterion(output, labels)
            loss = loss * masks
            loss = loss.mean()

            # Backward
            loss.backward()
            optimizer.step()

            iter_ += 1
            global_iter += 1

            summary_writer.add_scalar('Train/Loss', loss.item(), global_iter)
            logger.info('[{} {}/{} L:{}]'.format(epoch, iter_, len(dataloader), loss.item()))


def main():
    # Data 
    transforms_ = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CheXpertDataset(mode='train', transforms=transforms_)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.batch_size, 
                                  shuffle=True, 
                                  num_workers=config.num_workers, 
                                  pin_memory=True)

    model, optimizer, loss = experiment.get_experiment()

    summary_writer = tensorboardX.SummaryWriter(log_dir='runs/{}'.format(config.experiment_name))
    train_func(train_dataloader, model, optimizer, loss, device, summary_writer)


if __name__ == '__main__':
    if not config.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main()