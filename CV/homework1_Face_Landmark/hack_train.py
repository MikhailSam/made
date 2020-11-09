"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

from hack_utils import NUM_PTS, CROP_SIZE
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys, HFlip
from hack_utils import ThousandLandmarksDataset
from hack_utils import restore_landmarks_batch, create_submission

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

writer = SummaryWriter('drive/My Drive/CV/runs/resnext101')
NUM_WORKERS = 4
DEBUG = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=256, type=int)  # 512 is OK for resnet18 finetune @ 6Gb of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true", default=True)
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device, lr_scheduler=None):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        # # save image:
        # for i in range(len(images[0])):
        #     img = images[i]
        #     save_image(img, os.path.join(os.path.curdir, 'results', f'img{i}.jpg'))
        #     np.save(os.path.join(os.path.curdir, 'results', f'{i}.pkl'), landmarks[i].numpy())


        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        # flipped = batch["flipped"].numpy()
        # print('FLIPPED')
        # print(flipped)
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        # HFlip(),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
    ])

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="train")
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS,
                                       pin_memory=True, shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="val")
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS,
                                     pin_memory=True, shuffle=False, drop_last=False)

    print("Creating model...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    # model = models.resnet18(pretrained=True)
    model = models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    loss_fn = fnn.mse_loss
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device, lr_scheduler=scheduler)
        val_loss = validate(model, val_dataloader, loss_fn, device=device)
        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(f"drive/My Drive/CV/{args.name}_best.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'test'), train_transforms, split="test")
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS,
                                      pin_memory=True, shuffle=False, drop_last=False)

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    print(test_predictions)
    with open(f"drive/My Drive/CV/{args.name}_test_predictions.pkl", "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, f"drive/My Drive/CV/{args.name}_submit.csv")
    writer.close()


if __name__ == '__main__':
    if not DEBUG:
        args = parse_arguments()
    else:
        parser = ArgumentParser(__doc__)
        parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                            default="baseline")
        parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.",
                            default='../data/small/')
        parser.add_argument("--batch-size", "-b", default=8,
                            type=int)  # 512 is OK for resnet18 finetune @ 6Gb of VRAM
        parser.add_argument("--epochs", "-e", default=1, type=int)
        parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
        parser.add_argument("--gpu", action="store_true", default=True)
        args = parser.parse_args()
        print(args)
    sys.exit(main(args))
