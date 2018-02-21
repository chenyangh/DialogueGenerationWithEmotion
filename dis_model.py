"""
Copy from https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
Original author: Qinyuan Jiang, 2017
"""
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import os


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        # self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(context).unsqueeze(2)  # batch x dim x 1
        target = F.tanh(target)
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = F.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = F.relu(self.linear_out(h_tilde))

        return h_tilde, attn


class SelfAttention2 (nn.Module):
    """Soft Dot Attention.
    Ref: blablabla
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SelfAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.mask = None
        self.u_w = Variable(torch.randn(dim, 1)).cuda()

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        u = F.selu(self.linear_in(context))  # batch x dim x 1
        # u.view(u.size()[0] * u.size()[1], u.size()[2])
        # Get attention
        attn = F.softmax((u @ self.u_w).squeeze(2), dim=0).unsqueeze(1)
        h_tilde = torch.bmm(attn, context).squeeze(1)
        return h_tilde, attn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                     requires_grad=True)

        nn.init.xavier_uniform(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):

        batch_size, max_len = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).view(-1, 1).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word2id,
                 label_size, batch_size):
        super(AttentionLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.pad_token_src = word2id['<pad>']
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bidirectional = False
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_token_src)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        # self.hidden = self.init_hidden()
        if self.bidirectional:
            self.hidden2label = nn.Linear(hidden_dim*2, label_size)
            self.attention_layer = SelfAttention(hidden_dim*2)
        else:
            self.attention_layer = SelfAttention(hidden_dim)
            self.hidden2label = nn.Linear(hidden_dim, label_size)

        # self.last_layer = nn.Linear(hidden_dim, label_size * 100)
        # loss
        #weight_mask = torch.ones(vocab_size).cuda()
        #weight_mask[word2id['<pad>']] = 0
        # self.loss_criterion = nn.BCELoss()

    def init_hidden(self, x):
        batch_size = x.size(0)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad=False).cuda()
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=False).cuda()
        return (h0, c0)

    def forward(self, x, seq_len):
        embedded = self.embeddings(x)

        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, seq_len.numpy(), batch_first=True)
        hidden = self.init_hidden(x)
        packed_output, hidden = self.lstm(packed_input, hidden)
        lstm_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # global attention
        if False:
            output = lstm_out
            seq_len = torch.LongTensor(unpacked_len).view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            seq_len = Variable(seq_len - 1).cuda()
            output_extracted = torch.gather(output, 1, seq_len).squeeze(1)
            y_pred = F.relu(self.hidden2label(output_extracted))  # lstm_out[:, -1:].squeeze(1)
        else:
            # output = lstm_out
            # seq_len = torch.LongTensor(unpacked_len).view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            # seq_len = Variable(seq_len - 1).cuda()
            # output_extracted = torch.gather(output, 1, seq_len).squeeze(1) # lstm_out[:, -1:].squeeze(1)
            # # out, att = self.attention_layer(output_extracted, lstm_out)
            out, att = self.attention_layer(lstm_out, unpacked_len)
            y_pred = F.relu(self.hidden2label(out))
        # loss = self.loss_criterion(nn.Sigmoid()(y_pred), y)

        return y_pred

    def load_glove_embedding(self, id2word):
        """
        :param id2word:
        :return:
        """
        emb = np.zeros((self.vocab_size, self.embedding_dim))
        with open('feature/glove.twitter.200d.pkl', 'br') as f:
            emb_dict = pickle.load(f)
        num_found = 0
        for idx in range(self.vocab_size):
            word = id2word[idx]
            if word == '<pad>':
                emb[idx] = np.zeros([self.embedding_dim])

            elif word in emb_dict:
                vec = emb_dict[word]
                emb[idx] = vec
                num_found += 1
            else:
                emb[idx] = np.random.uniform(-1, 1, self.embedding_dim)
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        print(num_found, 'of', self.vocab_size, 'found')

    def load_bog_embedding(self, word2id):
        """"
        :param word2id:
        :return:
        """
        # Create BOW embedding
        emb = np.eye(self.vocab_size)
        emb[word2id['<pad>']] = np.zeros([self.vocab_size])
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        self.embeddings.weight.requires_grad = False
