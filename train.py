import pandas as pd
import numpy as np
import json
import random
import os
import sys
import pickle
from easydict import EasyDict as edict
import time
from datetime import datetime
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW,get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

import config as train_config
from dataset import get_dataloader
from util import save_checkpoint, one_hot, iter_product, clip_gradient, load_model
from sklearn.metrics import accuracy_score, f1_score
import loss as loss
from model import primary_encoder, weighting_network
import math


def train(epoch, train_loader, model_main, model_helper, loss_function, optimizer, lr_scheduler,log):

    model_main.cuda()
    model_main.train()

    model_helper.cuda()
    model_helper.train()

    total_true, total_pred_1, total_pred_2, acc_curve_1, acc_curve_2 = [], [], [], [], []
    train_loss_1, train_loss_2 = 0, 0
    total_epoch_acc_1, total_epoch_acc_2 = 0, 0
    steps = 0
    start_train_time = time.time()

    if log.param.is_waug is True:
        train_batch_size = log.param.batch_size * 2
    else:
        train_batch_size = log.param.batch_size
    for idx, batch in enumerate(train_loader):
        text_name = "sentence"
        label_name = "label"

        text = batch[text_name]
        attn = batch[text_name+"_attn_mask"]
        label = batch[label_name]
        label = torch.tensor(label)
        label = torch.autograd.Variable(label).long()

        if (label.size()[0] is not train_batch_size):# Last batch may have length different than log.param.batch_size
            continue

        if torch.cuda.is_available():
            text = text.cuda()
            attn = attn.cuda()
            label = label.cuda()

        pred_1, supcon_feature_1 = model_main(text, attn)
        pred_2 = model_helper(text, attn)
        pred_1_again, _ = model_main(text, attn)

        if log.param.loss_type == "lcl_drop":
            loss_1 = (loss_function["lambda_loss"] * loss_function["R_Drop"](pred_1, pred_1_again, label)) + ((1 - loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature_1, label, pred_2))
        elif log.param.loss_type == "lcl":
            loss_1 = (loss_function["lambda_loss"] * loss_function["ce"](pred_1, label)) + ((1 - loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature_1, label, pred_2))
        elif log.param.loss_type == "scl":
            loss_1 = (loss_function["lambda_loss"] * loss_function["ce"](pred_1, label)) + ((1 - loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature_1, label))
        elif log.param.loss_type == "scl_drop":
            loss_1 = (loss_function["lambda_loss"] * loss_function["R_Drop"](pred_1, pred_1_again, label)) + \
                     ((1 - loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature_1, label))
        else:
            loss_1 = loss_function["ce"](pred_1, label)
        loss_2 = loss_function["lambda_loss"] * loss_function["ce"](pred_2, label)

        if log.param.loss_type == "lcl" or log.param.loss_type == "lcl_drop":
            loss = loss_1 + loss_2
        else:
            loss = loss_1
        train_loss_1 += loss_1.item()
        train_loss_2 += loss_2.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(model_helper.parameters(), max_norm=1.0)
        optimizer.step()
        model_main.zero_grad()
        model_helper.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print(f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss_1: {loss_1.item():.4f}, Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()

        true_list = label.data.detach().cpu().tolist()
        total_true.extend(true_list)

        num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()
        pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
        total_pred_1.extend(pred_list_1)

        acc_1 = 100.0 * (num_corrects_1/train_batch_size)
        acc_curve_1.append(acc_1.item())
        total_epoch_acc_1 += acc_1.item()

        num_corrects_2 = (torch.max(pred_2, 1)[1].view(label.size()).data == label.data).float().sum()
        pred_list_2 = torch.max(pred_2, 1)[1].view(label.size()).data.detach().cpu().tolist()
        total_pred_2.extend(pred_list_2)

        acc_2 = 100.0 * (num_corrects_2/train_batch_size)
        acc_curve_2.append(acc_2.item())
        total_epoch_acc_2 += acc_2.item()

    return train_loss_1/len(train_loader), train_loss_2/len(train_loader), total_epoch_acc_1/len(train_loader), total_epoch_acc_2/len(train_loader), acc_curve_1, acc_curve_2


def test(epoch, test_loader, model_main, model_helper, loss_function, log):
    model_main.eval()
    model_helper.eval()
    test_loss = 0
    total_epoch_acc_1, total_epoch_acc_2 = 0, 0
    total_pred_1, total_pred_2, total_true, total_pred_prob_1, total_pred_prob_2 = [], [], [], [], []
    save_pred = {"true":[], "pred_1":[], "pred_2":[], "pred_prob_1":[], "pred_prob_2":[], "feature":[]}
    acc_curve_1, acc_curve_2 = [], []
    total_feature = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            text_name = "sentence"
            label_name = "label"

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]
            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            pred_1, supcon_feature_1= model_main(text, attn)
            pred_2 = model_helper(text, attn)

            num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()
            pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            acc_1 = 100.0 * num_corrects_1/1
            acc_curve_1.append(acc_1.item())
            total_epoch_acc_1 += acc_1.item()

            num_corrects_2 = (torch.max(pred_2, 1)[1].view(label.size()).data == label.data).float().sum()
            pred_list_2 = torch.max(pred_2, 1)[1].view(label.size()).data.detach().cpu().tolist()

            total_pred_1.extend(pred_list_1)
            total_pred_2.extend(pred_list_2)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(pred_1.data.detach().cpu().tolist())
            total_pred_prob_2.extend(pred_2.data.detach().cpu().tolist())

            acc_2 = 100.0 * num_corrects_2/1
            acc_curve_2.append(acc_2.item())
            total_epoch_acc_2 += acc_2.item()

    f1_score_1 = f1_score(total_true, total_pred_1, average="macro")
    f1_score_2 = f1_score(total_true, total_pred_2, average="macro")

    f1_score_1_w = f1_score(total_true, total_pred_1, average="weighted")
    f1_score_2_w = f1_score(total_true, total_pred_2, average="weighted")

    f1_score_1 = {"macro": f1_score_1, "weighted": f1_score_1_w}
    f1_score_2 = {"macro": f1_score_2, "weighted": f1_score_2_w}

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1
    save_pred["pred_2"] = total_pred_2
    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1
    save_pred["pred_prob_2"] = total_pred_prob_2

    return total_epoch_acc_1/len(test_loader), total_epoch_acc_2/len(test_loader), f1_score_1, f1_score_2, save_pred,\
           acc_curve_1, acc_curve_2


def lcl_train(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    train_data, valid_data, test_data = get_dataloader(log.param.batch_size, log.param.corpus, log.param.stego_method,
                                                       log.param.dataset, w_aug=log.param.is_waug)

    if log.param.loss_type == "lcl" or log.param.loss_type == "lcl_drop":
        losses = {"contrastive": loss.LCL(temperature=log.param.temperature), "ce": nn.CrossEntropyLoss(),
                  "lambda_loss":log.param.lambda_loss, "R_Drop": loss.RDrop()}
    elif log.param.loss_type == "scl" or log.param.loss_type == "scl_drop":
        losses = {"contrastive": loss.SupConLoss(temperature=log.param.temperature), "ce": nn.CrossEntropyLoss(),
                  "lambda_loss":log.param.lambda_loss, "R_Drop": loss.RDrop()}
    else:
        losses = {"ce": nn.CrossEntropyLoss(), "lambda_loss":log.param.lambda_loss,
                  "contrastive":loss.SupConLoss(temperature=log.param.temperature), "R_Drop": loss.RDrop()}
    train_loss_overall, test_loss_overall, train_accuracy_overall, test_accuracy_overall = 0, 0, 0, 0

    run_start = datetime.now()
    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model_main = primary_encoder(log.param.batch_size, log.param.hidden_size, log.param.label_size, log.param.model_type)
    model_helper = weighting_network(log.param.batch_size, log.param.hidden_size, log.param.label_size)
    total_params = list(model_main.named_parameters()) + list(model_helper.named_parameters())

    num_training_steps = int(len(train_data) * log.param.nepoch)
    print("num_training_steps: ", num_training_steps)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)], 'weight_decay': log.param.decay},
    {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    save_home = "./save/final/"+log.param.corpus+"/"+log.param.stego_method+"/"+log.param.dataset+"/"+log.param.loss_type+"/"+model_run_time+"/"
    #save_home = "./save/final/"+log.param.dataset+"/"+log.param.loss_type+"/"+model_run_time+"/"
    total_train_acc_curve_1, total_train_acc_curve_2, total_val_acc_curve_1, total_val_acc_curve_2 = [], [], [], []

    for epoch in range(1, log.param.nepoch + 1):
        train_loss_1, train_loss_2, train_acc_1, train_acc_2, train_acc_curve_1, train_acc_curve_2 = train(epoch, train_data, model_main, model_helper, losses, optimizer, lr_scheduler, log)
        val_acc_1, val_acc_2, val_f1_1, val_f1_2, val_save_pred, val_acc_curve_1, val_acc_curve_2 = test(epoch, valid_data, model_main, model_helper, losses, log)
        test_acc_1, test_acc_2, test_f1_1, test_f1_2, test_save_pred, test_acc_curve_1, test_acc_curve_2 = test(epoch, test_data, model_main, model_helper, losses, log)

        total_train_acc_curve_1.extend(train_acc_curve_1)
        total_val_acc_curve_1.extend(val_acc_curve_1)
        total_train_acc_curve_2.extend(train_acc_curve_2)
        total_val_acc_curve_2.extend(val_acc_curve_2)

        print('====> Epoch: {} Train loss_1: {:.4f}'.format(epoch, train_loss_1))
        os.makedirs(save_home, exist_ok=True)
        with open(save_home + "/acc_curve.json", 'w') as fp:
            json.dump({"train_acc_curve_1": total_train_acc_curve_1, "val_acc_curve_1": total_val_acc_curve_1}, fp, indent=4)
        fp.close()

        if epoch == 1:
             best_criterion = 0
        # is_best = val_acc_1 > best_criterion
        # best_criterion = max(val_acc_1, best_criterion)
        is_best = test_acc_1 > best_criterion
        best_criterion = max(test_acc_1, best_criterion)

        print("Model 1")
        print(f'Valid Accuracy: {val_acc_1:.2f}  Valid F1: {val_f1_1["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc_1:.2f}  Test F1: {test_f1_1["macro"]:.2f}')

        print("Model 2")
        print(f'Valid Accuracy: {val_acc_2:.2f}  Valid F1: {val_f1_2["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc_2:.2f}  Test F1: {test_f1_2["macro"]:.2f}')

        if is_best:
            print("======> Best epoch <======")
            patience_flag = 0
            log.train_loss_1 = train_loss_1
            log.train_loss_2 = train_loss_2
            log.stop_epoch = epoch
            log.stop_step = len(total_val_acc_curve_2)
            log.valid_f1_score_1 = val_f1_1
            log.test_f1_score_1 = test_f1_1
            log.valid_accuracy_1 = val_acc_1
            log.test_accuracy_1 = test_acc_1
            log.train_accuracy_1 = train_acc_1

            log.valid_f1_score_2 = val_f1_2
            log.test_f1_score_2 = test_f1_2
            log.valid_accuracy_2 = val_acc_2
            log.test_accuracy_2 = test_acc_2
            log.train_accuracy_2 = train_acc_2

            ## save the model
            # torch.save(model_main.state_dict(), save_home+'best.pt')
            run_end = datetime.now()
            best_time = str((run_end - run_start).seconds / 60) + ' minutes'
            log.best_time = best_time

            ## load the model
            with open(save_home + "/log.json", 'w') as fp:
                json.dump(dict(log), fp, indent=4)
            fp.close()

            with open(save_home + "/feature.json", 'w') as fp:
                json.dump(test_save_pred, fp, indent=4)
            fp.close()


if __name__ == '__main__':

    tuning_param = train_config.tuning_param
    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list))  # [(param_name),(param combinations)]

    for param_com in param_list[1:]:  # as first element is just name
        log = edict()
        log.param = train_config.param

        for num, val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        # reseeding before every run while tuning

        if "bpw" in log.param.dataset:
            log.param.label_size = 2
        elif "allbpw" in log.param.dataset:
            log.param.label_size = 6

        for loss_type in ['lcl', 'lcl_drop', 'scl_drop']:
            log.param.loss_type = loss_type
            run_start = datetime.now()
            lcl_train(log)
            run_end = datetime.now()
            run_time = str((run_end - run_start).seconds / 60) + ' minutes'
            print("corpus: ", log.param.corpus, "stego_method: ", log.param.stego_method,
                "dataset: ", log.param.dataset, "model_type: ", log.param.model_type, "run_time: ", run_time)