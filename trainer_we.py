import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from model import AttentionLSTMClassifier
from torch.utils.data import Dataset, DataLoader
from early_stop import EarlyStop
from measurement import CalculateFM
import numpy as np
import matplotlib.pyplot as plt


class DataSet(Dataset):
    def __init__(self, __fold_path, __pad_len, __word2id, __num_labels, max_size=None):

        self.pad_len = __pad_len
        self.word2id = __word2id
        self.pad_int = __word2id['<pad>']
        if max_size is not None:
            self.source = self.source[:max_size]
            self.target = self.target[:max_size]
            self.tag = self.tag[:max_size]
        self.data = []
        self.label = []
        self.num_label = __num_labels
        self.seq_len = []
        self.read_data(__fold_path)
        assert len(self.seq_len) == len(self.data) == len(self.label)

    def read_data(self, __fold_path):
        with open(__fold_path, 'r') as f:
            for line in f.readlines():
                tokens = line.split('\t')
                tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens[1].split()]
                self.seq_len.append(len(tmp) if len(tmp) < self.pad_len else self.pad_len)
                if len(tmp) > self.pad_len:
                    tmp = tmp[: self.pad_len]
                self.data.append(tmp + [self.pad_int] * (self.pad_len - len(tmp)))
                tmp2 = tokens[2:]
                a_label = [0] * self.num_label
                for item in tmp2:
                    a_label[int(item)] = 1
                self.label.append(a_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.seq_len[idx]]), torch.FloatTensor(self.label[idx])


def build_vocab(fold_path, vocab_size, use_unk=True):
    word_count = {}
    word2id = {}
    id2word = {}
    with open(os.path.join(fold_path, 'train.csv'), 'r') as f:
        for line in f.readlines():
            tokens = line.split('\t')
            sent = tokens[1]
            for word in sent.split():
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        word_list = [x for x, _ in sorted(word_count.items(), key=lambda v: v[1], reverse=True)]
        if len(word_count) < vocab_size:
            raise Exception('Vocab less than requested!!!')

        # add <pad> first
        word2id['<pad>'] = 0
        id2word[0] = '<pad>'
        if use_unk:
            word2id['<unk>'] = 1
            id2word[1] = '<unk>'
        n = len(word2id)
        word_list = word_list[:vocab_size - n]

        for word in word_list:
            word2id[word] = n
            id2word[n] = word
            n += 1
    return word2id, id2word


def sort_batch(batch, ys, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    targ_tensor = ys[perm_idx]
    return seq_tensor, targ_tensor, seq_lengths


def one_fold(fold_int, is_nine_folds):
    fold_id = str(fold_int)
    if is_nine_folds:
        fold_path = 'data/Folds_9_Emotions/fold_' + fold_id
        num_labels = 9
    else:
        fold_path = 'data/Folds/fold_' + fold_id
        num_labels = 16

    vocab_size = 5000
    pad_len = 30
    batch_size = 64
    embedding_dim = 200
    hidden_dim = 600

    es = EarlyStop(2)
    word2id, id2word = build_vocab(fold_path, vocab_size, use_unk=True)
    train_data = DataSet(os.path.join(fold_path, 'train.csv'), pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = DataSet(os.path.join(fold_path, 'test.csv'), pad_len, word2id, num_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size)
    model.load_glove_embedding(id2word)
    model.cuda()

    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.BCELoss()
    for epoch in range(4):
        print('Epoch:', epoch, '===================================')
        train_loss = 0
        for i, (data, seq_len, label) in enumerate(train_loader):
            data, label, seq_len = sort_batch(data, label, seq_len.view(-1))
            y_pred = model(Variable(data).cuda(), seq_len)
            optimizer.zero_grad()
            loss = loss_criterion(y_pred, Variable(label).cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        pred_list = []
        gold_list = []
        test_loss = 0
        for i, (data, seq_len, label) in enumerate(test_loader):
            data, label, seq_len = sort_batch(data, label, seq_len.view(-1))
            y_pred = model(Variable(data, volatile=True).cuda(), seq_len)
            loss = loss_criterion(y_pred, Variable(label, volatile=True).cuda())
            test_loss += loss.data[0]
            pred_list.append(y_pred.data.cpu().numpy())
            gold_list.append(label.numpy())

        print("Train Loss: ", train_loss, " Evaluation: ", test_loss)
        es.new_loss(test_loss)
        if es.if_stop():
            print('Start over fitting')
            break
    f_ma = []
    f_mi = []
    for threshold in range(0, 100, 5):
        threshold /= 100
        tmp = CalculateFM(np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0), threshold=threshold)
        f_ma.append(tmp['MacroFM'])
        f_mi.append(tmp['MicroFM'])
    return f_ma, f_mi


if __name__ == '__main__':
    f_ma_list = []
    f_mi_list = []
    for i in range(5):
        f_ma, f_mi = one_fold(i, is_nine_folds=True)
        f_ma_list.append(f_ma)
        f_mi_list.append(f_mi)

    f_ma_np_9 = np.asarray(f_ma_list).mean(axis=0)
    f_mi_np_9 = np.asarray(f_mi_list).mean(axis=0)

    f_ma_list = []
    f_mi_list = []
    for i in range(5):
        f_ma, f_mi = one_fold(i, is_nine_folds=False)
        f_ma_list.append(f_ma)
        f_mi_list.append(f_mi)

    f_ma_np_16 = np.asarray(f_ma_list).mean(axis=0)
    f_mi_np_16 = np.asarray(f_mi_list).mean(axis=0)

    import scipy.io as sio

    sio.savemat('we.mat', {'we_9_ma': f_ma_np_9,
                            'we_9_mi': f_mi_np_9,
                            'we_16_ma': f_ma_np_16,
                            'we_16_mi': f_mi_np_16})

