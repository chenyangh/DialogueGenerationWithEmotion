"""Decode Seq2Seq model with beam search."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.dec_rep import PersonaSeq2SeqAttentionSharedEmbedding
from utils.beam_search import Beam
import pandas as pd
import numpy as np
from jiwei_dataset import build_dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
NUM_EMO = 9

class BeamSearchDecoder(object):
    """Beam Search decoder."""

    def __init__(
        self,
        _model,
        _data_loader,
        _pad_len,
        _beam_size=3,
        _word2id=None,
        _id2word=None
    ):
        """Initialize model."""
        # self.config = config
        self.model = _model
        self.beam_size = _beam_size
        self.data_loader = _data_loader
        self.src_dict = _word2id
        self.tgt_dict = _word2id
        self.pad_len = _pad_len
        self.id2word = _id2word

    def get_hidden_representation(self, input):
        """Get hidden representation for a sentence."""
        src_emb = self.model.embedding(input)
        h0_encoder, c0_encoder = self.model.get_state(src_emb)
        src_h, (src_h_t, src_c_t) = self.model.encoder(
            src_emb, (h0_encoder, c0_encoder)
        )

        if self.model.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        return src_h, (h_t, c_t)

    def get_init_state_decoder(self, input):
        """Get init state for decoder."""
        decoder_init_state = nn.Tanh()(self.model.encoder2decoder(input))
        return decoder_init_state

    def decode_batch(self, input_lines_src, _emo):
        """Decode a minibatch."""
        # Get source minibatch

        beam_size = self.beam_size

        #  (1) run the encoder on the src

        context_h, (context_h_t, context_c_t) = self.get_hidden_representation(
            input_lines_src
        )

        context_h = context_h.transpose(0, 1)  # Make things sequence first.

        #  (3) run the decoder to generate sentences, using beam search

        batch_size = context_h.size(1)

        # Expand tensors for each beam.
        context = Variable(context_h.data.repeat(1, beam_size, 1))
        dec_states = [
            Variable(context_h_t.data.repeat(1, beam_size, 1)),
            Variable(context_c_t.data.repeat(1, beam_size, 1))
        ]

        beam = [
            Beam(beam_size, self.tgt_dict, cuda=True)
            for k in range(batch_size)
        ]

        dec_out = self.get_init_state_decoder(dec_states[0].squeeze(0))
        dec_states[0] = dec_out

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.pad_len):

            input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            trg_emb = self.model.embedding(Variable(input).transpose(1, 0))
            tag = [_emo] * trg_emb.size(0)
            tag = Variable(torch.LongTensor(tag).cuda())
            trg_h, (trg_h_t, trg_c_t) = self.model.decoder(
                trg_emb, tag,
                (dec_states[0].squeeze(0), dec_states[1].squeeze(0)),
                context
            )

            dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1)
            out = F.softmax(self.model.decoder2vocab(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.model.decoder.hidden_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores

    def translate(self, _emo):
        """Translate the whole dataset."""
        trg_preds = []
        trg_gold = []

        for i, (src, trg, _) in tqdm(enumerate(self.data_loader),
                                     total=(len(self.data_loader.dataset)/self.data_loader.batch_size)):

            """Decode a single minibatch."""
            # print('Decoding %d out of %d ' % (j, len(self.src['data'])))
            hypotheses, scores = self.decode_batch(Variable(src, volatile=True).cuda(), _emo)
            all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
            all_preds = [
                ' '.join([self.id2word[x] for x in hyp])
                for hyp in all_hyp_inds
            ]
            # input_lines_trg_gold = src
            output_lines_trg_gold = Variable(trg, volatile=True).cuda()
            # # Get target minibatch
            # input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            #     get_minibatch_no_tag(
            #         self.trg['data'], self.tgt_dict, j,
            #         self.config['data']['batch_size'],
            #         self.config['data']['max_trg_length'],
            #         add_start=True, add_end=True
            #     )
            # )

            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
            all_gold = [
                ' '.join([self.id2word[x] for x in hyp])
                for hyp in all_gold_inds
            ]
            trg_preds += all_preds
            trg_gold += all_gold

        df = pd.DataFrame({'preds': [''.join(pred) for pred in trg_preds],
                           'gold': [''.join(ground_truth) for ground_truth in trg_gold]
                        })
        df.to_csv('outputs/persona_beam' + str(self.beam_size) + '_' + str(_emo) + '.csv', encoding='utf-8', index=False)

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


def main(beam_size):

    word2id, id2word = build_dict()
    pad_len = 30
    batch_size = 500
    emb_dim = 300
    dim = 600
    vocab_size = len(word2id)

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
    ).cuda()

    model_path = 'checkpoint/new_persona_epoch_22.model'

    model.load_state_dict(torch.load(
        model_path
    ))

    df = pd.read_csv('data_6_remove_dup_test.csv')
    X, y, tag = df['source'], df['target'], df['tag']

    test_set = EmotionDataLoader(X, y, tag, pad_len, word2id)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    for __emo in range(9):
        decoder = BeamSearchDecoder(model, test_loader, pad_len, beam_size, word2id, id2word)
        decoder.translate(__emo)

        # decoder = GreedyDecoder(config, model_weights, src_test, trg_test, word2id=src['word2id'])
        # out = decoder.translate(emo)


for beam_size in range(2, 3):
    main(beam_size)

    '''
    allHyp, allScores = decoder.decode_batch(0)
    all_hyp_inds = [[x[0] for x in hyp] for hyp in allHyp]
    all_preds = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_hyp_inds]

    input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
        get_minibatch(
            trg['data'], trg['word2id'], 0,
            80,
            50,
            add_start=True, add_end=True
        )
    )

    output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
    all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
    all_gold = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_gold_inds]

    for hyp, gt in zip(all_preds, all_gold):
        print hyp, len(hyp.split())
        print '-------------------------------------------------'
        print gt
        print '================================================='
    '''
