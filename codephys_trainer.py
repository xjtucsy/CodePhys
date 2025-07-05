import torch
import numpy as np
import os
import random
import time
import argparse
import julius
import logging
import shutil

from scipy import io as sio
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from archs.code_rppgnet import CodeRppgNet
from archs.physnet_divided import VideoFeatureExtractor, WeightMapGenerator, RppgEstimator, PPGFeatureExtractor
from datasets.rppg_datasets import VIPL, UBFC, PURE, COHFACE
from losses.NPLoss import Neg_Pearson
from losses.CELoss import TorchLossComputer
from losses.CodeLoss import code_loss
from losses.FeatureLoss import FeatureLossComputerSpatial
from utils.utils import AvgrageMeter, cxcorr_align

def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def _init_fn(seed=92):
    np.random.seed(seed)

class CodephysTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.video_feature_extractor = VideoFeatureExtractor().to(self.device)
        self.rppg_estimator = RppgEstimator(frames=args.num_rppg).to(self.device)
        self.codebook_module = CodeRppgNet(args).to(self.device)
        self.weight_map_generator = WeightMapGenerator().to(self.device)
        self.codebook_module_blocked = CodeRppgNet(args).to(self.device)
        self.ppg_feature_extractor = PPGFeatureExtractor(args).to(self.device)

        ## generate save path
          
        self.save_path = f'{args.save_path}/{self.run_date}'
        
        ## dataloader NOTE: SELECT YOUR DATASET
        # -------------------------------------------------------
        ## VIPL dataset
        if args.dataset == 'VIPL':
            self.train_dataset = VIPL(data_dir=args.dataset_dir, T=args.num_rppg, train='train', w=128, h=128, fold=1)
        # -------------------------------------------------------
        ## UBFC dataset
        elif args.dataset == 'UBFC':
            self.train_dataset = UBFC(data_dir=args.dataset_dir, T=args.num_rppg, train='train', w=128, h=128)
        # -------------------------------------------------------
        ## PURE dataset
        elif args.dataset == 'PURE':
            self.train_dataset = PURE(data_dir=args.dataset_dir, T=args.num_rppg, train='train', w=128, h=128)
        # -------------------------------------------------------
        ## COHFACE dataset
        elif args.dataset == 'COHFACE':
            self.train_dataset = COHFACE(data_dir=args.dataset_dir, T=args.num_rppg, train='train', w=128, h=128)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=_init_fn)

        ## optimizer
        self.optimizer = torch.optim.Adam(params=list(self.video_feature_extractor.parameters()) + \
                                        list(self.ppg_feature_extractor.parameters()) + \
                                        list(self.weight_map_generator.parameters()) + \
                                        list(self.rppg_estimator.parameters()) + \
                                        list(self.codebook_module.encoder.parameters()) + \
                                        list(self.codebook_module.rppg_local_stream.parameters()) + \
                                        list(self.codebook_module.local_global_mixer.parameters()) , \
                                        lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.criterion_Pearson = Neg_Pearson()
        self.feature_loss = FeatureLossComputerSpatial()

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        self.frame_rate = 30        
    
    def prepare_train(self, start_epoch, continue_log):
        if start_epoch != 0:
            self.save_path = self.args.save_path + '/' + continue_log
            self.run_date = continue_log

        self.save_ckpt_path = f'{self.save_path}/ckpt'
        self.save_rppg_path = f'{self.save_path}/rppg'
        if not os.path.exists(self.save_ckpt_path):
            os.makedirs(self.save_ckpt_path)
        if not os.path.exists(self.save_rppg_path):
            os.makedirs(self.save_rppg_path)
        
        logging.basicConfig(filename=f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}.log',\
                            format='%(message)s', filemode='a')
        self.logger = logging.getLogger(f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}')
        self.logger.setLevel(logging.INFO)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split('/')[-1]
        shutil.copytree(f'{cur_file}', f'{self.save_path}/{cur_file_name}', dirs_exist_ok=True)

        if start_epoch != 0:
            if not os.path.exists(f'{self.save_ckpt_path}/video_feature_extractor_{start_epoch - 1}.pth'):
                raise Exception(f'video_feature_extractor ckpt file {start_epoch - 1} not found')
            if not os.path.exists(f'{self.save_ckpt_path}/rppg_estimator_{start_epoch - 1}.pth'):
                raise Exception(f'rppg_estimator ckpt file {start_epoch - 1} not found')
            if not os.path.exists(f'{self.save_ckpt_path}/codebook_{start_epoch - 1}.pth'):
                raise Exception(f'codebook ckpt file {start_epoch - 1} not found')
            self.video_feature_extractor.load_state_dict(torch.load(f'{self.save_ckpt_path}/video_feature_extractor_{start_epoch - 1}.pth'))
            self.rppg_estimator.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_{start_epoch - 1}.pth'))
            self.codebook_module.load_state_dict(torch.load(f'{self.save_ckpt_path}/codebook_{start_epoch - 1}.pth'))
            self.codebook_module_blocked.load_state_dict(torch.load(f'{self.save_ckpt_path}/codebook_{start_epoch - 1}.pth'))
            self.weight_map_generator.load_state_dict(torch.load(f'{self.save_ckpt_path}/weight_map_generator_{start_epoch - 1}.pth'))
            self.ppg_feature_extractor.load_state_dict(torch.load(f'{self.save_ckpt_path}/ppg_feature_extractor_{start_epoch - 1}.pth'))
        else:
            self.codebook_module.load_state_dict(torch.load(f'{self.args.codebook_model_path}', map_location=self.device))
            self.codebook_module_blocked.load_state_dict(torch.load(f'{self.args.codebook_model_path}', map_location=self.device))
    
        ## block gradient for codebook module (codebook and decoder)
        self.video_feature_extractor.train()
        self.ppg_feature_extractor.train()
        self.weight_map_generator.train()
        self.rppg_estimator.train()
        self.codebook_module.train()
        self.codebook_module_blocked.eval()
        for name, param in self.codebook_module.named_parameters():
            if name.split('.')[0] in ['recover_embed', 'decoder_token_mlp', 'decoder_start', 'decoder', 'codebook']:
                param.requires_grad = False
    
    def draw_rppg_ecg(self, rPPG, ecg, save_path_epoch, train=False, mini_batch=0):
        ## save the results
        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
        # 使用 lfilter 函数进行滤波
        rPPG_np = rPPG[0].cpu().data.numpy()
        rPPG_np = signal.lfilter(b, a, rPPG_np)
        rPPG[0] = torch.from_numpy(rPPG_np).to(self.device)
        results_rPPG = []
        y1 = rPPG[0].cpu().data.numpy()
        y2 = ecg[0].cpu().data.numpy() 
        results_rPPG.append(y1)
        results_rPPG.append(y2)
        if not train:
            sio.savemat(os.path.join(save_path_epoch, f'test_rPPG.mat'), {'results_rPPG': results_rPPG})
        else:
            sio.savemat(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.mat'), {'results_rPPG': results_rPPG})
        # show the ecg images
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        psd_pred = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, self.bpm_range)
        psd_gt = TorchLossComputer.complex_absolute(ecg[0], self.frame_rate, self.bpm_range)
        ax[0].set_title('rPPG')
        ax[0].plot(y1, label='rPPG')
        ax[0].plot(y2, label='ecg')
        ax[0].legend()
        ax[1].set_title('psd')
        ax[1].plot(psd_pred[0].cpu().data.numpy(), label='pred')
        ax[1].plot(psd_gt[0].cpu().data.numpy(), label='gt')
        ax[1].legend()
        if not train:
            fig.savefig(os.path.join(save_path_epoch, f'test_rPPG.jpg'))
        else:
            fig.savefig(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.jpg'))
        plt.close(fig)

    def train(self, start_epoch=0, continue_log=''):

        self.prepare_train(start_epoch=start_epoch, continue_log=continue_log)
        self.logger.info(f'prepare train, load ckpt and block gradient, start_epoch: {start_epoch}')

        echo_batches = self.args.echo_batches
        gamma = self.args.gamma
        step_size = self.args.step_size
        eval_step = self.args.eval_step
        lr = self.args.lr * (gamma ** (start_epoch // step_size))
        optimizer = self.optimizer
        scheduler = self.scheduler

        for epoch in range(start_epoch, self.args.epochs):
            if epoch % step_size == step_size - 1:
                lr *= gamma
            loss_rPPG_avg = AvgrageMeter()
            loss_peak_avg = AvgrageMeter()
            loss_kl_avg = AvgrageMeter()
            loss_hr_mae = AvgrageMeter()
            loss_code_avg = AvgrageMeter()
            loss_e_latent = AvgrageMeter()
            loss_align_avg = AvgrageMeter()
            loss_ppg_elatent = AvgrageMeter()
            loss_ppg_code_avg = AvgrageMeter()
            save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
            if not os.path.exists(save_path_epoch):
                os.makedirs(save_path_epoch)
            self.logger.info(f'train epoch: {epoch} lr: {lr}')
            with tqdm(range(len(self.train_dataloader))) as pbar:
                for i, sample_batched in zip(pbar, self.train_dataloader):
                    inputs, ecg = sample_batched['video'].to(self.device), sample_batched['ecg'].to(self.device)
                    clip_average_HR  = sample_batched['clip_avg_hr'].to(self.device)

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    
                    ## for backbone divieded:
                    video_feature = self.video_feature_extractor(inputs)            # extract video feature
                    ppg_feature = self.ppg_feature_extractor(ecg)                   # extract ppg feature
                    spatial_weight_map = self.weight_map_generator(video_feature)   # generate spatial weight map => [B, H, W]
                    mask_video_feature = video_feature * spatial_weight_map.unsqueeze(1).unsqueeze(1)        # mask video feature, feed to dual-stream network
                    rPPG_peak = self.rppg_estimator(mask_video_feature)            # global stream middle signal
                    rPPG = (rPPG_peak - rPPG_peak.mean(dim=-1, keepdim=True)) / \
                                torch.abs(rPPG_peak).max(dim=-1, keepdim=True).values  # normalize
                    

                    ppg_feature_signal = self.rppg_estimator(ppg_feature)          # global stream middle signal for ppg feature
                    ppg_feature_signal = (ppg_feature_signal - ppg_feature_signal.mean(dim=-1, keepdim=True)) / \
                                torch.abs(ppg_feature_signal).max(dim=-1, keepdim=True).values  # normalize
                    ppg_feature_signal, ppg_feature_indices, ppg_feature_latent_loss = \
                        self.codebook_module(ppg_feature_signal, ppg_feature)  # local-stream & code query & decoder for ppg feature

                    rPPG, encoding_indices, e_latent_loss = self.codebook_module(rPPG, mask_video_feature)   # local-stream & code query & decoder

                    ## calculate loss
                    rPPG = cxcorr_align(rPPG, ecg)  # align
                    loss_rPPG = self.criterion_Pearson(rPPG, ecg)    # np_loss
                    loss_code = code_loss(encoding_indices, ecg, self.codebook_module_blocked)  # code_loss
                    loss_code_ppg_feature = code_loss(ppg_feature_indices, ecg, self.codebook_module_blocked)  # code_loss for ppg feature
                    loss_align = self.feature_loss(video_feature, ppg_feature.detach(), spatial_weight_map.detach())    # align_loss

                    clip_average_HR = (clip_average_HR - 40)    # for 'TorchLossComputer.cross_entropy_power_spectrum_loss',\
                                                                # the input HR should be in range [0, 140] instead of [40, 180]

                    fre_loss = 0.0
                    kl_loss = 0.0
                    train_mae = 0.0
                    for bb in range(inputs.shape[0]):
                        kl_loss_temp, fre_loss_temp, train_mae_temp = \
                              TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(rPPG[bb],\
                                                            clip_average_HR[bb], self.frame_rate, 1.0)
                        fre_loss = fre_loss + fre_loss_temp
                        kl_loss = kl_loss + kl_loss_temp
                        train_mae = train_mae + train_mae_temp
                    fre_loss = fre_loss / inputs.shape[0]
                    kl_loss = kl_loss / inputs.shape[0]
                    train_mae = train_mae / inputs.shape[0]

                    loss = loss_align + 2.0 * (0.01 * loss_rPPG + fre_loss + kl_loss) + 0.1 * (e_latent_loss + loss_code + loss_code_ppg_feature + ppg_feature_latent_loss)

                    loss.backward()
                    optimizer.step()

                    ## update loss saver
                    n = inputs.size(0)
                    loss_rPPG_avg.update(loss_rPPG.data, n)
                    loss_peak_avg.update(fre_loss.data, n)
                    loss_kl_avg.update(kl_loss.data, n)
                    loss_hr_mae.update(train_mae.data, n)
                    loss_code_avg.update(loss_code.data, n)
                    loss_e_latent.update(e_latent_loss.data, n)
                    loss_align_avg.update(loss_align.data, n)
                    loss_ppg_elatent.update(ppg_feature_latent_loss.data, n)
                    loss_ppg_code_avg.update(loss_code_ppg_feature.data, n)

                    if i % echo_batches == echo_batches - 1:  # info every mini-batches
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, NP_loss = {loss_rPPG_avg.avg:.4f}, ' \
                            f'fre_loss = {loss_peak_avg.avg:.4f}, kl_loss = {loss_kl_avg.avg:.4f}, hr_mae = {loss_hr_mae.avg[0]:.2f}, ' \
                            f'code_loss = {loss_code_avg.avg:.4f}, e_latent_loss = {loss_e_latent.avg:.4f}, align_loss = {loss_align_avg.avg:.4f}, ' \
                                f'ppg_e_latent_loss = {loss_ppg_elatent.avg:.4f}, ppg_code_loss = {loss_ppg_code_avg.avg:.4f}')
                        # save the ecg images
                        self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=i)

                    pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.4f}, NP_loss = {loss_rPPG_avg.avg:.3f}, ' \
                            f'fre_loss = {loss_peak_avg.avg:.3f}, kl_loss = {loss_kl_avg.avg:.3f}, hr_mae = {loss_hr_mae.avg[0]:.2f}, ' \
                            f'code_loss = {loss_code_avg.avg:.2f}, e_latent_loss = {loss_e_latent.avg:.2f}, align_loss = {loss_align_avg.avg:.2f}, ' \
                                f'ppg_e_latent_loss = {loss_ppg_elatent.avg:.2f}, ppg_code_loss = {loss_ppg_code_avg.avg:.2f}')                        
                    
            scheduler.step()

            # save the model
            torch.save(self.codebook_module.state_dict(), os.path.join(self.save_ckpt_path, f'codebook_{epoch}.pth'))
            torch.save(self.video_feature_extractor.state_dict(), os.path.join(self.save_ckpt_path, f'video_feature_extractor_{epoch}.pth'))
            torch.save(self.rppg_estimator.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_{epoch}.pth'))
            torch.save(self.weight_map_generator.state_dict(), os.path.join(self.save_ckpt_path, f'weight_map_generator_{epoch}.pth'))
            torch.save(self.ppg_feature_extractor.state_dict(), os.path.join(self.save_ckpt_path, f'ppg_feature_extractor_{epoch}.pth'))

            # delete the model
            if epoch > 0:
                os.remove(os.path.join(self.save_ckpt_path, f'codebook_{epoch-1}.pth'))
                os.remove(os.path.join(self.save_ckpt_path, f'video_feature_extractor_{epoch-1}.pth'))
                os.remove(os.path.join(self.save_ckpt_path, f'rppg_estimator_{epoch-1}.pth'))
                os.remove(os.path.join(self.save_ckpt_path, f'weight_map_generator_{epoch-1}.pth'))
                os.remove(os.path.join(self.save_ckpt_path, f'ppg_feature_extractor_{epoch-1}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--gpu', type=int, default=2, help='gpu id')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='the drop rate of CodeResNet')
    parser.add_argument('--train_stage', type=str, default='train_codephys', help='train_stage = [train_codebook, train_codephys]')
    parser.add_argument('--dataset', type=str, default='VIPL')
    parser.add_argument('--dataset_dir', type=str, default='', help='dataset dir')

    ### add for codephys
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--echo_batches', type=int, default=500, help='the number of mini-batches to print the loss')
    parser.add_argument('--save_path', type=str, default='', help='the path to save the model [ckpt, code, visulization]')

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
    parser.add_argument('--ada_hidden_dim', type=int, default=64)
    parser.add_argument('--ada_hidden_inner_dim', type=int, default=128)
    parser.add_argument('--ada_num_layers', type=int, default=2)
    parser.add_argument('--ada_dropout_ratio', type=float, default=0.0)

    ## codebook parameters
    parser.add_argument('--vq_token_dim', type=int, default=64)
    parser.add_argument('--token_class_num', type=int, default=64)
    parser.add_argument('--fmix_rate', type=float, default=1.0, help='fmix rate should be 0 when training codebook')
    parser.add_argument('--codebook_model_path', type=str, default="", help='the path to load the codebook model')

    args = parser.parse_args()

    set_seed(92)

    codephys_trainer = CodephysTrainer(args)
    codephys_trainer.train(start_epoch=0, continue_log='0811_1250') # NOTE: WHETHER TO CONTINUE TRAINING
