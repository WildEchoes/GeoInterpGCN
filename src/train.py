import os
import argparse
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.dataset import GeoDataset
from src.model import GeoInterpGCN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", type=str, help="points data file's location")
    parser.add_argument("--label", type=str, help="which element need to be trained")

    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--train_rate", type=float, default=0.8, help="train rate")

    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=3, help="random seed")
    parser.add_argument("--num_workers", type=int, default=11, help="num_workers")

    option = parser.parse_args()

    # time_flag = datetime.now().strftime("%Y%m%d-%H%M%S")
    # option.timeflag = time_flag

    return option


def main(opt):
    print("----------------------")
    print(f"{opt.label} Train Start!")
    save_path = os.path.join("./model", f"{opt.timeflag}", f"{opt.label}")
    os.makedirs(save_path, exist_ok=True)

    model_name = os.path.join(save_path, f"{opt.label}.pth")
    params_name = os.path.join(save_path, "train_params.pth")
    log_file = open(os.path.join(save_path, f"log.txt"), "w")

    torch.save(opt, params_name)
    print("----------------------")
    print(str(opt))
    print("----------------------")
    log_file.write(str(opt) + "\n")

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    print("Initnializing Dataset...")
    datalist: list = GeoDataset(opt.datafile, opt.label).get

    train_dataset, test_dataset= train_test_split(datalist, train_size=opt.train_rate, random_state=opt.seed)

    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    print("Initnializing Model...")
    model = GeoInterpGCN(input_dim=4, output_dim=1)

    if torch.cuda.is_available():
        model.cuda(opt.gpu)
    else:
        print("CUDA is not available!")

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=4, factor=0.5, min_lr=1e-8)

    best_loss = float('inf')
    # time_start = datetime.now()
    torch.cuda.memory_reserved()

    for epoch in range(0, opt.epochs):
        # ---------- Train ------------
        model.train()
        desc = "TRAINING - epoch:{:>3} loss:{:.4f}"
        bar = "{desc}  ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}]"
        pbar = tqdm(total=len(train_loader), ncols=90, leave=False, desc=desc.format(epoch, 0), bar_format=bar)

        optimizer.zero_grad()
        all_train_loss = []
        for step, data in enumerate(train_loader):
            # iteration = len(train_loader) * (epoch-1) + step
            data_ = data.cuda(opt.gpu)
            out = model(data_)
            loss = torch.nn.MSELoss()(out, data_.y)
            
            all_train_loss.append(loss.item())
            
            loss /= opt.batchsize
            loss.backward()
            loss *= opt.batchsize

            if (((step + 1) % opt.batchsize) == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

                pbar.desc = desc.format(epoch, loss)
                pbar.update()
        pbar.close()

        # ---------- Test ------------
        all_test_loss = []
        model.eval()
        with torch.no_grad():
            desc = "VALIDATION - epoch:{:>3} loss:{:.4f}"
            pbar = tqdm(total=len(test_loader), ncols=90, leave=False, desc=desc.format(epoch, 0), bar_format=bar)

            for i, data in enumerate(test_loader):
                data_ = data.cuda(opt.gpu)
                out = model(data_)
                loss = torch.nn.MSELoss()(out, data_.y)
                all_test_loss.append(loss.item())

                pbar.desc = desc.format(epoch, loss)
                pbar.update()
        pbar.close()

        # ----- Update lr --------
        train_loss = np.array(all_train_loss).mean()
        test_loss = np.array(all_test_loss).mean()
        scheduler.step(train_loss)
        info = f"epoch: {epoch:>3} Train loss: {train_loss:>7.4f} | Test loss:{test_loss:>7.4f} "
        print(info)
        log_file.write(info + "\n")

        # span = datetime.now() - time_start

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), model_name)
            print(f"{epoch} model saved!------")
            log_file.write(f"{epoch} model saved!------\n")

    log_file.close()


if __name__ == "__main__":
    opt = parse_args()
    opt.datafile = "E:/WorkSpace/GeoInterpGCN/data/chem_dem.csv"

    time_flag = datetime.now().strftime("%Y%m%d-%H%M%S")
    opt.timeflag = time_flag
    labels = ["Al2O3", "Fe2O3", "K2O", "MgO", "Na2O", "SiO2", "Cu"]
    for label in labels:
        opt.label = label
        main(opt)