import torch
import numpy as np
import random
import os
import math
import argparse
import logging

from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import  DataLoader

from archs.code_rppgnet import CodeRppgNet
from datasets.signal_datasets import VIPL, UBFC, PURE, COHFACE
from losses.NPLoss import Neg_Pearson
from losses.CELoss import TorchLossComputer

def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn(seed=92):
    np.random.seed(seed)

class Codebook_trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        ## VIPL dataset
        if args.dataset == 'VIPL':
            self.train_dataset = VIPL(data_dir=args.dataset_dir, train=True, rppg_len=args.num_rppg, fold=5)
            self.val_dataset = VIPL(data_dir=args.dataset_dir, train=False, rppg_len=args.num_rppg, fold=5)
        ## UBFC dataset
        elif args.dataset == 'UBFC':
            self.train_dataset = UBFC(data_dir=args.dataset_dir, train=True, rppg_len=args.num_rppg)
            self.val_dataset = UBFC(data_dir=args.dataset_dir, train=False, rppg_len=args.num_rppg)
        ## PURE dataset
        elif args.dataset == 'PURE':
            self.train_dataset = PURE(data_dir=args.dataset_dir, train=True, rppg_len=args.num_rppg)
            self.val_dataset = PURE(data_dir=args.dataset_dir, train=False, rppg_len=args.num_rppg)
        ## COHFACE dataset
        elif args.dataset == 'COHFACE':
            self.train_dataset = COHFACE(data_dir=args.dataset_dir, train=True, rppg_len=args.num_rppg)
            self.val_dataset = COHFACE(data_dir=args.dataset_dir, train=False, rppg_len=args.num_rppg)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=_init_fn)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=_init_fn)

        self.model = CodeRppgNet(args).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.loss_np = Neg_Pearson()

        if not os.path.exists(args.save_root):
            os.makedirs(args.save_root)

        logging.basicConfig(filename=f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}.log',\
                            format='%(message)s', filemode='w')
        self.logger = logging.getLogger(f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}')
        self.logger.setLevel(logging.INFO)

    def val(self, epoch):
        self.model.eval()
        np_loss_lst = []
        mse_loss_lst = []
        codebook_loss_lst = []
        with torch.no_grad():
            for rppg in self.val_dataloader:
                rppg = rppg.to(self.device)
                recon, code, codebook_loss = self.model(rppg)
                codebook_loss_lst.append(codebook_loss.item())
                np_loss = self.loss_np(recon, rppg)
                np_loss_lst.append(np_loss.item())
                mse_loss = torch.nn.MSELoss()(recon, rppg)
                mse_loss_lst.append(mse_loss.item())

        Fs = 30
        bpm_range = torch.arange(40, 180, dtype=torch.float)
        psd = TorchLossComputer.complex_absolute(rppg[0], Fs, bpm_range)
        psd_recon = TorchLossComputer.complex_absolute(recon[0], Fs, bpm_range)

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(rppg[0].cpu().numpy(), label="rppg", color="red", linewidth=2)
        ax[0].plot(recon[0].cpu().numpy(), label="recon", linewidth=2)
        ax[0].legend()
        ax[1].plot(psd[0].detach().cpu().numpy(), label="rppg", linewidth=2)
        ax[1].plot(psd_recon[0].detach().cpu().numpy(), label="recon", linewidth=2)
        ax[1].legend()
        fig.savefig(f"{args.save_root}/{epoch}_info.jpg")
        plt.close()

        self.logger.info(f"val np loss: {np.mean(np_loss_lst)}, mse loss: {np.mean(mse_loss_lst)}, codebook loss: {np.mean(codebook_loss_lst)}")

    def train(self, train_dataloader, epoch_num, model_path):
        self.model.train()
        codebook_loss_lst = []
        mse_loss_lst = []
        total_loss_lst = []
        for epoch in range(epoch_num):
            self.logger.info(f'training on epoch {epoch}...')
            codebook_loss_epoch = 0
            mse_loss_epoch = 0
            total_loss_epoch = 0
            with tqdm(range(len(train_dataloader))) as pbar:
                for i, rppg in zip(pbar, train_dataloader):
                    rppg = rppg.to(self.device)
                    recon, code, codebook_loss = self.model(rppg)
                    
                    mse_loss = torch.nn.MSELoss()(recon, rppg)
                    # mse_loss = torch.Tensor([0]).to(self.device)
                    np_loss = self.loss_np(recon, rppg)
                    loss =  codebook_loss +  mse_loss + np_loss

                    total_loss_epoch += loss.item()
                    codebook_loss_epoch += codebook_loss.item()
                    mse_loss_epoch += mse_loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_description(f'loss: {loss.item():.4f}, codebook_loss: {codebook_loss.item():.4f}, mse_loss: {mse_loss.item():.4f}')
                                         
            self.val(epoch)

            codebook_loss_lst.append(codebook_loss_epoch / len(train_dataloader))
            mse_loss_lst.append(mse_loss_epoch / len(train_dataloader))
            total_loss_lst.append(total_loss_epoch / len(train_dataloader))
        ## save model
        self.logger.info(f'saving model to {model_path}...')
        torch.save(self.model.state_dict(), model_path)

        ## plot loss
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(codebook_loss_lst, label="codebook_loss", linewidth=2)
        ax.plot(mse_loss_lst, label="mse_loss", linewidth=2)
        ax.plot(total_loss_lst, label="total_loss", linewidth=2)
        ax.legend()
        fig.savefig(f"{self.args.save_root}/loss.jpg")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160, help='number of rppg frames')
    parser.add_argument('--gpu', type=int, default=7, help='gpu')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate for CodeRppgNet')
    parser.add_argument('--train_stage', type=str, default='train_codebook', help='train_codebook or train_codephys')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-3)

    parser.add_argument('--epoch_num', type=int, default=15)
    parser.add_argument('--save_root', type=str, default='', help='root for saving model and loss')
    parser.add_argument('--dataset', type=str, default='COHFACE', help='VIPL or PURE or UBFC')
    parser.add_argument('--dataset_dir', type=str, default='', help='dataset dir')

    ## encoder parameters
    parser.add_argument('--enc_hidden_dim', type=int, default=512)
    parser.add_argument('--enc_hidden_inner_dim', type=int, default=512)
    parser.add_argument('--enc_num_layers', type=int, default=4)
    parser.add_argument('--enc_dropout_ratio', type=float, default=0.0)
    
    ## decoder parameters
    parser.add_argument('--dec_hidden_dim', type=int, default=32)
    parser.add_argument('--dec_hidden_inner_dim', type=int, default=64)
    parser.add_argument('--dec_num_layers', type=int, default=1)
    parser.add_argument('--dec_dropout_ratio', type=float, default=0.0)

    ## adapter parameters
    parser.add_argument('--ada_hidden_dim', type=int, default=64, help='equal to vq_token_dim')
    parser.add_argument('--ada_hidden_inner_dim', type=int, default=128)
    parser.add_argument('--ada_num_layers', type=int, default=2)
    parser.add_argument('--ada_dropout_ratio', type=float, default=0.0)

    ## codebook parameters
    parser.add_argument('--vq_token_dim', type=int, default=64)
    parser.add_argument('--token_class_num', type=int, default=64)
    parser.add_argument('--fmix_rate', type=float, default=0.0, help='fmix rate for training codephys, must be 0 for training codebook')

    args = parser.parse_args()

    set_seed(92)

    trainer = Codebook_trainer(args)
    trainer.train(trainer.train_dataloader, args.epoch_num, f'{args.save_root}/model.pth')
