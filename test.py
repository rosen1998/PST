from data import PST_DATA
from PSTNet import PST
from run import test_PST as test

import argparse
import torch
import numpy as np
import os
import pickle
import logging

target_type = {
    'is_ac': 'main',
    'first_rate': 'downstream',
    'avg_rate': 'downstream',
    'last_rate': 'downstream'
}

def set_random_seed(args):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


def read_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def test_one_dataset(test_data, logger, target, args):
    model = PST(args.seq_len, args.num_exercises, args.embedding_dim, args.device, args.max_position, args.dropout if args.dropout > 0 else 0.2)
    logger.info(model)
    print(model)
    D = PST_DATA(args.seq_len)
    test_targets, test_pro_ids, test_detail_is_acs, test_rates = D.load_data(test_data, target)
    path = os.path.join(args.model_path, args.dataset_name, args.model_name+'_'+target if target_type[target] == 'downstream' else args.model_name+'_main')
    test_e_cig = torch.load(os.path.join(args.embedding_path, args.dataset_name, 'test-cig.pt')).to(args.device)
    test_e_ctg = torch.load(os.path.join(args.embedding_path, args.dataset_name, 'test-ctg.pt')).to(args.device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info('test pharse')
    with torch.no_grad():
        test_performance, test_loss = test(model, test_targets, test_e_cig, test_e_ctg, test_detail_is_acs, test_rates, test_pro_ids, logger, args, target)
    print(test_performance)
    logger.info('test performance: {0}'.format(test_performance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_exercises', type=int, default=1671)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--embedding_path', type=str, default='embedding')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--dataset_name', type=str, default='atcoder_c')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--train_name', type=str, default='train.pkl')
    parser.add_argument('--val_name', type=str, default='val.pkl')
    parser.add_argument('--test_name', type=str, default='test.pkl')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--max_position', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()
    print(args)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_random_seed(args)
    test_data_dir = os.path.join(args.data_dir, args.dataset_name, args.test_name)

    print('getting test data...')
    test_data = read_pkl(test_data_dir)


    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)

    for target in ['is_ac', 'first_rate', 'last_rate', 'avg_rate']:
        if os.path.exists(os.path.join(args.log_dir, args.dataset_name, args.model_name+'_'+target+'_test.txt')):
            os.remove(os.path.join(args.log_dir, args.dataset_name, args.model_name+'_'+target+'_test.txt'))
        handler = logging.FileHandler(os.path.join(args.log_dir, args.dataset_name, args.model_name+'_'+target+'_test.txt'))
        logger.addHandler(handler)
        test_one_dataset(test_data, logger, target, args)

        if handler:
            logger.removeHandler(handler)