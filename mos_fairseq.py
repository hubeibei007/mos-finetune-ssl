# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
random.seed(1984)

import numpy as np
import scipy.stats
import csv

from sodeep_master.sodeep import load_sorter, SpearmanLoss

def save_results(ep, valid_result, test_result, result_path):
    if os.path.isfile(result_path):
        with open(result_path, "r", newline='') as csvfile:
            rows = list(csv.reader(csvfile))
        data = {row[0]: row[1:] for row in rows}
    else:
        data = {}
    data[str(ep)] = valid_result + test_result
    rows = [[k]+v for k, v in data.items()]
    rows = sorted(rows, key=lambda x:int(x[0]))
    with open(result_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.output_layer(x)
        return x.squeeze(1)

    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        return wav, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames

def lcc_loss_fn(outputs, labels):
    #void division by zero
    if outputs.std() == 0.0:
        print('std outputs zero', outputs.std())
    if labels.std() == 0.0:
        print('std labels zero', labels.std())

    vo = outputs - torch.mean(outputs)
    vl = labels - torch.mean(labels)
    batch_size = outputs.shape[0]
    eps = 1e-9 #void /0.0
    lcc = torch.sum(vo*vl)/(outputs.std()*labels.std() + eps)/(batch_size-1)

    loss = -1 * lcc
    return loss

def validation(net, validloader, epoch, global_step, device):
    #valset
    net.eval()
    ## clear memory to avoid OOM
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    ## validation
    with torch.no_grad():
        print('Starting validation')
        filename_list = []
        pred_score_list = []
        gt_score_list = []
        for i, data in enumerate(validloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            pred_scores = outputs.cpu().detach().numpy()
            true_scores = labels.cpu().detach().numpy()

            filename_list.extend(filenames)
            pred_score_list.extend(pred_scores.tolist())
            gt_score_list.extend(true_scores.tolist())

        #calc pred sys scores and gt sys scores
        predict_scores = np.array(pred_score_list)
        true_scores = np.array(gt_score_list)
        systems = list(set([x.split('-')[0] for x in filename_list]))
        predict_sys_scores = {system:[] for system in systems}
        true_sys_scores = {system:[] for system in systems}
        for i in range(len(filename_list)):
            filename = filename_list[i]
            system = filename.split('-')[0]
            pre_score = predict_scores[i]
            gt_score = true_scores[i]
            predict_sys_scores[system].append(pre_score)
            true_sys_scores[system].append(gt_score)
        predict_sys_scores = np.array([np.mean(scores) for scores in predict_sys_scores.values()])
        true_sys_scores = np.array([np.mean(scores) for scores in true_sys_scores.values()])

        utt_MSE=np.mean((true_scores-predict_scores)**2)
        utt_LCC=np.corrcoef(true_scores, predict_scores)[0][1]
        utt_SRCC=scipy.stats.spearmanr(true_scores, predict_scores)[0]
        utt_KTAU=scipy.stats.kendalltau(true_scores, predict_scores)[0]
        sys_MSE=np.mean((true_sys_scores-predict_sys_scores)**2)
        sys_LCC=np.corrcoef(true_sys_scores, predict_sys_scores)[0][1]
        sys_SRCC=scipy.stats.spearmanr(true_sys_scores, predict_sys_scores)[0]
        sys_KTAU=scipy.stats.kendalltau(true_sys_scores, predict_sys_scores)[0]

        net.train()
        print(f"\n[{epoch}][UTT][ MSE = {utt_MSE:.4f} | LCC = {utt_LCC:.4f} | SRCC = {utt_SRCC:.4f} ] [SYS][ MSE = {sys_MSE:.4f} | LCC = {sys_LCC:.4f} | SRCC = {sys_SRCC:.4f} ]\n")
        return utt_MSE, utt_LCC, utt_SRCC, utt_KTAU, sys_MSE, sys_LCC, sys_SRCC, sys_KTAU

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--sorter_checkpoint', type=str, required=False, help='Path to SoDeep pretrained checkpoint')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    sorter_checkpoint_path = args.sorter_checkpoint

    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    
    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn, drop_last=True)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion_mae = nn.L1Loss()
    criterion_spr = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    criterion_spr.to(device)
    criterion_lcc = lcc_loss_fn

    optimizer = optim.SGD(list(net.parameters())+list(criterion_spr.parameters()),  lr=0.0001, momentum=0.9)

    global_step = 0
    for epoch in range(1, 301):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss_mae = criterion_mae(outputs, labels)
            loss_spr = criterion_spr(outputs, labels)
            loss_lcc = criterion_lcc(outputs, labels)
            loss = loss_mae + loss_spr + loss_lcc

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            STEPS += 1
            global_step += 1

        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))

        ## validation
        utt_MSE, utt_LCC, utt_SRCC, utt_KTAU,\
        sys_MSE, sys_LCC, sys_SRCC, sys_KTAU = validation(net, validloader, epoch, global_step, device)

        #recording data
        save_results(epoch,
                 [utt_MSE, utt_LCC, utt_SRCC, utt_KTAU], [sys_MSE, sys_LCC, sys_SRCC, sys_KTAU],
                 os.path.join(ckptdir, "training_val" + ".csv"))

        #save every epoch
        PATH = os.path.join(ckptdir, 'ckpt_' + str(epoch) + '_' +  str(global_step))
        torch.save(net.state_dict(), PATH)

    print('Finished Training')

if __name__ == '__main__':
    main()
