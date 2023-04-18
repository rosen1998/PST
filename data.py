import numpy as np
import dgl
import torch
from tqdm import tqdm
import math

class PST_DATA():
    def __init__(self, seq_len):
        self.seq_len = seq_len
    
    def load_data(self, data, target):
        pred_target = []
        pro_ids = []
        detail_is_ac = []
        rates = []
        for learning_seq in tqdm(data, desc='loading data...'):
            pred_target_arr = []
            pro_ids_arr = []
            detail_is_ac_arr = []
            rates_arr = []
            for learning_item in learning_seq:
                item_len = len(learning_item['is_ac_arr'])
                target_item = learning_item[target]
                now_target = [target_item] + [-1]*(item_len-1)
                pred_target_arr += now_target
                pro_ids_arr += [learning_item['pro_id']]*item_len
                detail_is_ac_arr += learning_item['is_ac_arr']
                rates_arr += learning_item['rate']
            n_split = 1
            if len(pred_target_arr)>self.seq_len:
                n_split = math.floor(len(pred_target_arr)/self.seq_len)
                if len(pred_target_arr)/self.seq_len:
                    n_split += 1
            for k in range(n_split):
                pred_target_seq = []
                pro_ids_seq = []
                detail_is_ac_seq = []
                rates_seq = []
                if k == n_split-1:
                    end_index = len(pred_target_arr)
                else:
                    end_index = (k+1)*self.seq_len
                for idx in range(k*self.seq_len, end_index):
                    pred_target_seq.append(pred_target_arr[idx])
                    pro_ids_seq.append(pro_ids_arr[idx])
                    detail_is_ac_seq.append(detail_is_ac_arr[idx])
                    rates_seq.append(rates_arr[idx])
                pred_target.append(pred_target_seq)
                pro_ids.append(pro_ids_seq)
                detail_is_ac.append(detail_is_ac_seq)
                rates.append(rates_seq)

        pred_target_np = np.zeros((len(pred_target), self.seq_len))
        pro_ids_np = np.zeros((len(pred_target), self.seq_len))
        detail_is_ac_np = np.zeros((len(pred_target), self.seq_len))
        rates_np = np.zeros((len(pred_target), self.seq_len))
        rates_np -= 1
        pred_target_np -= 1
        for i in tqdm(range(len(pred_target)), desc='get numpy...'):
            pred_target_np[i, :len(pred_target[i])] = pred_target[i]
            pro_ids_np[i, :len(pred_target[i])] = pro_ids[i]
            detail_is_ac_np[i, :len(pred_target[i])] = detail_is_ac[i]
            rates_np[i, :len(pred_target[i])] = rates[i]
        return pred_target_np, pro_ids_np, detail_is_ac_np, rates_np