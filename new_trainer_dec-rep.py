import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from jiwei_dataset import build_dict
from model3 import PersonaSeq2SeqAttentionSharedEmbedding
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from early_stop import EarlyStop

NUM_EMO = 9


class EmotionDataLoader(Dataset):
    def __init__(self, X, y, tag, pad_len, word2int, max_size=None):
        self.source = X
        self.target = y
        self.tag = tag
        self.pad_len = pad_len
        self.start_int = word2int['<s>']
        self.eos_int = word2int['</s>']
        self.pad_int = word2int['<pad>']
        self.word2id = word2int
        assert len(self.tag) == len(self.source)
        if max_size is not None:
            self.source = self.source[:max_size]
            self.target = self.target[:max_size]
            self.tag = self.tag[:max_size]

    def __len__(self):
        return len(self.tag)

    def __getitem__(self, idx):
        # for src add <s> ahead
        src = [self.word2id[x] for x in self.source[idx].split()]
        if len(src) > self.pad_len:
            src = src[:self.pad_len]
        src = src + [self.pad_int] * (self.pad_len - len(src))

        # for trg add <s> ahead and </s> end
        trg = [self.word2id[x] for x in self.target[idx].split()]
        if len(trg) > self.pad_len - 2:
            trg = trg[:self.pad_len-2]
        trg = [self.start_int] + trg + [self.eos_int] + [self.pad_int] * (self.pad_len - len(trg) - 2)
        if not len(src) == len(trg) == self.pad_len:
            print(src, trg)
        assert len(src) == len(trg) == self.pad_len
        if self.tag[idx] == 'Nan':
            tag = NUM_EMO
        else:
            tag = int(self.tag[idx])
        return torch.LongTensor(src), torch.LongTensor(trg), torch.LongTensor([tag])


if __name__ == '__main__':
    word2id, id2word = build_dict()
    pad_len = 30
    batch_size = 600
    emb_dim = 300
    dim = 600
    vocab_size = len(word2id)

    from sklearn.model_selection import ShuffleSplit
    split_ratio = 0.05
    sss = ShuffleSplit(n_splits=1, test_size=split_ratio, random_state=0)
    df = pd.read_csv('data_6_remove_dup_train.csv')
    X, y, tag = df['source'], df['target'], df['tag']
    # temp_up = 10000
    # X = X[:temp_up]
    # y = y[:temp_up]
    # tag = tag[:temp_up]

    train_index, dev_index = next(sss.split(tag))

    X_train = [X[i] for i in train_index]
    y_train = [y[i] for i in train_index]
    tag_train = [tag[i] for i in train_index]
    X_dev = [X[i] for i in dev_index]
    y_dev = [y[i] for i in dev_index]
    tag_dev = [tag[i] for i in dev_index]
    del df, X, y, tag

    training_set = EmotionDataLoader(X_train, y_train, tag_train, pad_len, word2id)
    train_loader = DataLoader(training_set, batch_size=batch_size)

    test_set = EmotionDataLoader(X_dev, y_dev, tag_dev, pad_len, word2id)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    # loader = iter(train_loader)
    # next(loader)

    model = PersonaSeq2SeqAttentionSharedEmbedding(
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        src_hidden_dim=dim,
        trg_hidden_dim=dim,
        ctx_hidden_dim=dim,
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=False,
        pad_token_src=word2id['<pad>'],
        pad_token_trg=word2id['<pad>'],
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
    )
    model.load_word_embedding(id2word)
    model.cuda()


    # model.load_state_dict(torch.load(
    #     'checkpoint/new_persona_epoch_7.model'
    # ))
    #

    weight_mask = torch.ones(vocab_size).cuda()
    weight_mask[word2id['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    es = EarlyStop(2)
    for epoch in range(100):
        print('Training on epoch=%d -------------------------' % (epoch))
        train_loss_sum = 0
        for i, (src, trg, tag) in tqdm(enumerate(train_loader), total=int(len(training_set)/batch_size)):
            # print('i=%d: ' % (i))
            decoder_logit = model(Variable(src).cuda(), Variable(trg).cuda(), Variable(tag.view(-1)).cuda())
            optimizer.zero_grad()
            trg = torch.cat((torch.index_select(trg, 1, torch.LongTensor(list(range(1, pad_len)))),
                             torch.LongTensor(np.zeros([trg.shape[0], 1]))), dim=1)
            loss = loss_criterion(
                decoder_logit.contiguous().view(-1, vocab_size),
                Variable(trg).view(-1).cuda()
            )
            train_loss_sum += loss.data[0]
            loss.backward()
            optimizer.step()
            del loss, decoder_logit
        print("Training Loss", train_loss_sum/len(training_set)*batch_size)

        # Evaluate
        test_loss_sum = 0
        print("Evaluating:")
        for i, (src_test, trg_test, tag_test) in tqdm(enumerate(test_loader), total=int(len(test_set)/batch_size)):

            test_logit = model(Variable(src_test, volatile=True).cuda(),
                               Variable(trg_test, volatile=True).cuda(),
                               Variable(tag_test.view(-1), volatile=True).cuda())
            trg_test = torch.cat((torch.index_select(trg_test, 1, torch.LongTensor(list(range(1, pad_len)))),
                             torch.LongTensor(np.zeros([trg_test.shape[0], 1]))), dim=1)
            test_loss = loss_criterion(
                test_logit.contiguous().view(-1, vocab_size),
                Variable(trg_test).view(-1).cuda()
            )
            test_loss_sum += test_loss.data[0]
            del test_loss, test_logit

        print("Evaluation Loss", test_loss_sum/len(test_set))
        es.new_loss(test_loss_sum)
        if es.if_stop():
            print('Start over fitting')
            break
        # Save Model
        torch.save(
            model.state_dict(),
            open(os.path.join(
                'checkpoint',
                'new_persona' + '_epoch_%d' % (epoch) + '.model'), 'wb'
            )
        )

