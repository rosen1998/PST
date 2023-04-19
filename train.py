from data import PST_DATA
from PSTNet import PST
from run import train_PST as train
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


def train_one_dataset(train_data, val_data, test_data, logger, target, args):
    model = PST(args.seq_len, args.num_exercises, args.embedding_dim, args.device, args.max_position, args.dropout if args.dropout > 0 else 0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(model)
    print(model)
    D = PST_DATA(args.seq_len)
    train_targets, train_pro_ids, train_detail_is_acs, train_rates = D.load_data(train_data, target)
    val_targets, val_pro_ids, val_detail_is_acs, val_rates = D.load_data(val_data, target)
    test_targets, test_pro_ids, test_detail_is_acs, test_rates = D.load_data(test_data, target)
    path = os.path.join(args.model_path, args.dataset_name, args.model_name+'_'+target if target_type[target] == 'downstream' else args.model_name+'_main')

    train_e_cig = torch.load(os.path.join(args.embedding_path, args.dataset_name, args.cig_train_embedding)).to(args.device)
    val_e_cig = torch.load(os.path.join(args.embedding_path, args.dataset_name, args.cig_val_embedding)).to(args.device)
    test_e_cig = torch.load(os.path.join(args.embedding_path, args.dataset_name, args.cig_test_embedding)).to(args.device)
    train_e_ctg = torch.load(os.path.join(args.embedding_path, args.dataset_name, args.ctg_train_embedding)).to(args.device)
    val_e_ctg = torch.load(os.path.join(args.embedding_path, args.dataset_name, args.ctg_val_embedding)).to(args.device)
    test_e_ctg = torch.load(os.path.join(args.embedding_path, args.dataset_name, args.ctg_test_embedding)).to(args.device)
    if target_type[target] == 'downstream':
        args.alpha = 1
        args.beta = 0
        main_model_path = os.path.join(args.model_path, args.dataset_name, args.model_name+'_main')
        checkpoint = torch.load(main_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        for p in model.parameters():
            p.requires_grad_(False)
        model.learning_fitting_encoder.W_10.requires_grad_(True)
        model.learning_fitting_encoder.W_11.requires_grad_(True)
        model.learning_fitting_encoder.W_12.requires_grad_(True)
        model.learning_fitting_encoder.W_13.requires_grad_(True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    best_loss = 10
    for idx in range(args.epoch):
        print('EPOCH {0} start:'.format(idx+1))
        logger.info('EPOCH {0} start:'.format(idx+1))
        logger.info('train pharse')
        train_loss = train(model, optimizer, train_targets, train_e_cig, train_e_ctg, train_detail_is_acs, train_rates, train_pro_ids, logger, args, target)
        logger.info('dev pharse')
        with torch.no_grad():
            val_performance, val_loss = test(model, val_targets, val_e_cig, val_e_ctg, val_detail_is_acs, val_rates, val_pro_ids, logger, args, target)
        print(val_performance)
        better_flag = val_loss<best_loss
        if better_flag:
            best_loss = val_loss
            torch.save({
                'epoch': idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                },
                path,
            )
        logger.info('train loss: {0}, val loss: {1}, val performance: {2}'.format(train_loss, val_loss, val_performance))
        print('train loss: {0}, val loss: {1}, val performance: {2}'.format(train_loss, val_loss, val_performance))
    logger.info('best loss: {0}'.format(best_loss))
    print('best loss: {0}'.format(best_loss))
    if args.do_test:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            test_performance, test_loss = test(model, test_targets, test_e_cig, test_e_ctg, test_detail_is_acs, test_rates, test_pro_ids, logger, args, target)
        print('test performance: {0}'.format(test_performance))
        logger.info('test performance: {0}'.format(test_performance))
        print('test performance: {0}'.format(test_performance))



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
    parser.add_argument('--cig_train_embedding', type=str, default='train-cig.pt')
    parser.add_argument('--cig_val_embedding', type=str, default='val-cig.pt')
    parser.add_argument('--cig_test-embedding', type=str, default='test-cig.pt')
    parser.add_argument('--ctg_train_embedding', type=str, default='train-ctg.pt')
    parser.add_argument('--ctg_val_embedding', type=str, default='val-ctg.pt')
    parser.add_argument('--ctg_test_embedding', type=str, default='test-ctg.pt')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--max_position', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--do_test', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_random_seed(args)
    train_data_dir = os.path.join(args.data_dir, args.dataset_name, args.train_name)
    val_data_dir = os.path.join(args.data_dir, args.dataset_name, args.val_name)
    test_data_dir = os.path.join(args.data_dir, args.dataset_name, args.test_name)

    print('getting train data...')
    train_data = read_pkl(train_data_dir)
    print('getting val data...')
    val_data = read_pkl(val_data_dir)
    print('getting test data...')
    test_data = read_pkl(test_data_dir)


    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)

    for target in ['is_ac', 'first_rate', 'last_rate', 'avg_rate']:
        if os.path.exists(os.path.join(args.log_dir, args.dataset_name, ''+args.model_name+'_'+target+'.txt')):
            os.remove(os.path.join(args.log_dir, args.dataset_name, ''+args.model_name+'_'+target+'.txt'))
        handler = logging.FileHandler(os.path.join(args.log_dir, args.dataset_name, ''+args.model_name+'_'+target+'.txt'))
        logger.addHandler(handler)
        train_one_dataset(train_data, val_data, test_data, logger, target, args)

        if handler:
            logger.removeHandler(handler)
