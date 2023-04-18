import torch
from sklearn import metrics
import math
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def RMSE(pred, y):
    return math.sqrt(metrics.mean_squared_error(y, pred))

def F1_score(pred, y):
    f1 = metrics.f1_score(y, pred)
    return f1

def auc(pred_score, true_y, label_value=1):
    f, t, _ = metrics.roc_curve(true_y, pred_score, pos_label=label_value)
    return metrics.auc(f, t)

def acc(pred, y):
    return metrics.accuracy_score(y, pred)

def train_PST(net, optimizer, targets, e_cig, e_ctg, detail_is_acs, rates, exercises, logger, args, optimize_target):
    net.to(args.device)
    net.train()
    N = int(math.ceil(len(targets) / args.batch_size))
    if 'rate' in optimize_target:
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.BCELoss()

    def loss_f(y, r, targets, rate_targets, args):
        loss1 = loss_func(y, targets)
        loss_func_two = nn.MSELoss()
        loss2 = loss_func_two(r, rate_targets)
        loss = args.alpha*loss1 + args.beta*loss2
        return loss

    pred_list = []
    target_list = []
    all_loss = 0
    pred_count = 0
    all_r = []
    all_rate = []
    
    for i in tqdm(range(N), desc='training a model...'):
        optimizer.zero_grad()
        target_batch = torch.from_numpy(targets[i*args.batch_size:(i+1)*args.batch_size])
        rates_batch = torch.from_numpy(rates[i*args.batch_size:(i+1)*args.batch_size])
        exercises_batch = np.array([])
        detail_is_ac_batch = np.array([])
        e_cig_batch = e_cig[i*args.batch_size:i*args.batch_size+target_batch.shape[0]].to(args.device)
        e_ctg_batch = e_ctg[i*args.batch_size:i*args.batch_size+target_batch.shape[0]].to(args.device)
        exercises_batch = torch.from_numpy(exercises[i*args.batch_size:(i+1)*args.batch_size]).long().to(args.device)
        detail_is_ac_batch = torch.from_numpy(detail_is_acs[i*args.batch_size:(i+1)*args.batch_size]).to(args.device)
        y_batch, r_batch = net(detail_is_ac_batch, exercises_batch, e_cig_batch, e_ctg_batch)
        y_batch = y_batch[:, 1:].cpu()
        r_batch = r_batch[:, 1:].cpu()
        target_batch = target_batch[:, 1:]
        target_batch = target_batch.reshape(-1)
        rates_batch = rates_batch[:, 1:].reshape(-1)
        y_batch = y_batch.reshape(-1)
        r_batch = r_batch.reshape(-1)
        # mask the pad data
        mask_count = torch.sum(target_batch >= -.9)
        pred_count += mask_count
        mask = target_batch >= -.9
        ra_mask = rates_batch >= -.9
        y_mask = y_batch[mask].double()
        target_mask = target_batch[mask].double()
        r_mask = r_batch[ra_mask].double()
        rate_mask = rates_batch[ra_mask].double()
        loss = loss_f(y_mask, r_mask, target_mask, rate_mask, args)


        # pass the pad data
        if torch.isnan(loss):
            pass
        else:
            all_loss += loss.item() * mask_count

        # update parameters
        loss.backward()
        optimizer.step()

        pred_list.append(y_mask.detach().numpy())
        target_list.append(target_mask.numpy())
        all_r.append(r_mask.detach().numpy())
        all_rate.append(rate_mask.numpy())

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    all_r = np.concatenate(all_r, axis=0)
    all_rate = np.concatenate(all_rate, axis=0)
    
    all_pred = torch.from_numpy(all_pred)
    all_target = torch.from_numpy(all_target)
    all_r = torch.from_numpy(all_r)
    all_rate = torch.from_numpy(all_rate)
    all_pred = all_pred.reshape(-1)
    all_target = all_target.reshape(-1)
    all_r = all_r.reshape(-1)
    all_rate = all_rate.reshape(-1)
    all_loss = loss_f(all_pred, all_r, all_target, all_rate, args)

    return all_loss

def test_PST(net, targets, e_cig, e_ctg, detail_is_acs, rates, exercises, logger, args, optimize_target):
    net.to(args.device)
    net.eval()
    N = int(math.ceil(len(targets) / args.batch_size))
    if 'rate' in optimize_target:
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.BCELoss()

    def loss_f(y, r, targets, rate_targets, args):
        loss1 = loss_func(y, targets)
        loss_func_two = nn.MSELoss()
        loss2 = loss_func_two(r, rate_targets)
        loss = args.alpha*loss1 + args.beta*loss2
        return loss

    pred_list = []
    target_list = []
    all_loss = 0
    pred_count = 0
    all_r = []
    all_rate = []
    
    for i in tqdm(range(N), desc='testing a model...'):
        target_batch = torch.from_numpy(targets[i*args.batch_size:(i+1)*args.batch_size])
        rates_batch = torch.from_numpy(rates[i*args.batch_size:(i+1)*args.batch_size])
        exercises_batch = np.array([])
        detail_is_ac_batch = np.array([])
        e_cig_batch = e_cig[i*args.batch_size:i*args.batch_size+target_batch.shape[0]].to(args.device)
        e_ctg_batch = e_ctg[i*args.batch_size:i*args.batch_size+target_batch.shape[0]].to(args.device)
        exercises_batch = torch.from_numpy(exercises[i*args.batch_size:(i+1)*args.batch_size]).long().to(args.device)
        detail_is_ac_batch = torch.from_numpy(detail_is_acs[i*args.batch_size:(i+1)*args.batch_size]).to(args.device)
        y_batch, r_batch = net(detail_is_ac_batch, exercises_batch, e_cig_batch, e_ctg_batch)
        y_batch = y_batch[:, 1:].cpu()
        r_batch = r_batch[:, 1:].cpu()
        target_batch = target_batch[:, 1:]
        target_batch = target_batch.reshape(-1)
        rates_batch = rates_batch[:, 1:].reshape(-1)
        y_batch = y_batch.reshape(-1)
        r_batch = r_batch.reshape(-1)
        # mask the pad data
        mask_count = torch.sum(target_batch >= -.9)
        pred_count += mask_count
        mask = target_batch >= -.9
        ra_mask = rates_batch >= -.9
        y_mask = y_batch[mask].double()
        target_mask = target_batch[mask].double()
        r_mask = r_batch[ra_mask].double()
        rate_mask = rates_batch[ra_mask].double()
        loss = loss_f(y_mask, r_mask, target_mask, rate_mask, args)


        # pass the pad data
        if torch.isnan(loss):
            pass
        else:
            all_loss += loss.item() * mask_count

        pred_list.append(y_mask.detach().numpy())
        target_list.append(target_mask.numpy())
        all_r.append(r_mask.detach().numpy())
        all_rate.append(rate_mask.numpy())

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    all_r = np.concatenate(all_r, axis=0)
    all_rate = np.concatenate(all_rate, axis=0)
    

    all_pred = torch.from_numpy(all_pred)
    all_target = torch.from_numpy(all_target)
    all_r = torch.from_numpy(all_r)
    all_rate = torch.from_numpy(all_rate)
    all_pred = all_pred.reshape(-1)
    all_target = all_target.reshape(-1)
    all_r = all_r.reshape(-1)
    all_rate = all_rate.reshape(-1)
    all_loss = loss_f(all_pred, all_r, all_target, all_rate, args)
    if 'rate' in optimize_target:
        rmse = RMSE(all_pred, all_target)
        performance = {
            'rmse': rmse
        }
    else:
        u = auc(all_pred, all_target)
        rmse = RMSE(all_pred, all_target)
        all_pred = np.round(all_pred)
        all_target = all_target
        c = acc(all_pred, all_target)
        f1 = F1_score(all_pred, all_target)
        performance = {
            'auc': u,
            'acc': c,
            'rmse': rmse,
            'f1': f1
        }

    print(performance)
    logger.info('performance: {0}'.format(performance))

    return performance, all_loss