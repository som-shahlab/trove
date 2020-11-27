import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from pytorch_pretrained_bert import BertModel


class TaggerBERT(nn.Module):
    """
    BERT-based tagger

    'BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding' (Devlin et al. 2019)
    https://arxiv.org/abs/1810.04805

    """
    def __init__(self,
                 num_classes,
                 bert_model='bert-base-cased',
                 hidden_size=768,
                 use_rnn=False,
                 rnn_dropout=0.1,
                 rnn_num_layers=2,
                 device='cpu',
                 finetuning=False):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model) if type(
            bert_model) is str else bert_model

        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = nn.LSTM(bidirectional=True,
                               num_layers=rnn_num_layers,
                               input_size=hidden_size,
                               hidden_size=hidden_size // 2,
                               batch_first=True,
                               dropout=rnn_dropout)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.finetuning = finetuning

    def _forward_rnn(self, X, lens):

        X_packed = rnn_utils.pack_padded_sequence(
            X, lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, c_n) = self.rnn(X_packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        return output

    def forward(self, xs, xidxs):

        xs = xs.to(self.device).long()

        if self.training and self.finetuning:
            self.bert.train()
            enc_layers, _ = self.bert(xs)
            enc_layers = enc_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                enc_layers, _ = self.bert(xs)
                enc_layers = enc_layers[-1]

        # restrict to head tokens
        if not self.use_subword_labels:
            enc = [layer[idxs] for layer, idxs in zip(enc_layers, xidxs)]
            x_lens = [layer.size(0) for layer in enc]

        # ignore [CLS] and [SEP] special tokens
        else:
            enc = [layer[1:-1] for layer in enc_layers]
            x_lens = [layer.size(0) for layer in enc]

        enc = rnn_utils.pad_sequence(enc, batch_first=True, padding_value=0)

        if self.use_rnn:
            enc = self._forward_rnn(enc, x_lens)

        return self.fc(enc)


class TaggerRNN(torch.nn.Module):
    """
    RNN-based tagger architecture as described in:

    'Neural Architectures for Named Entity Recognition' (Lample et al. 2016)
    https://arxiv.org/abs/1603.01360

    """
    def __init__(self,
                 hidden_size,
                 n_classes,
                 encoder,
                 num_layers=1,
                 dropout=0.5,
                 bidirectional=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.rnn = nn.LSTM(
            input_size=self.encoder.embed_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout  # isn't used when num_layers == 1
        )

        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear((2 if bidirectional else 1) * self.hidden_size,
                      self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, n_classes)
        )

    def _seq_lens(self, X):
        batch_size, max_seq = X.shape[0], X.shape[1]
        lens = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            for j in range(max_seq - 1, -1, -1):
                if not torch.all(X[i, j] == 0):
                    lens[i] = j + 1
                    break
        return lens

    def _forward_rnn(self, X):
        lens = self._seq_lens(X)
        X_packed = rnn_utils.pack_padded_sequence(
            self.encoder(X), lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, c_n) = self.rnn(X_packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        return output

    def _forward_rnn_ONNX(self, X):
        # ONNX-compliant pack_padded_sequence requires sorted seqs
        batch_size, max_seq = X.shape[0], X.shape[1]
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            for j in range(max_seq - 1, -1, -1):
                if not torch.all(X[i, j] == 0):
                    seq_lengths[i] = j + 1
                    break

        # sort by length for pack_padded_sequence
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx]
        inv_perm_idx = torch.tensor(
            [i for i, _ in
             sorted(enumerate(perm_idx), key=lambda idx: idx[1])],
            dtype=torch.long,
        )

        # pack and encode
        X_packed = rnn_utils.pack_padded_sequence(
            self.encoder(X), seq_lengths, batch_first=True
        )
        output, (h_n, c_n) = self.rnn(X_packed)

        # unpack and restore pre-sort order
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        return output[inv_perm_idx, :]

    def forward(self, X):
        output = self._forward_rnn(X[0])
        return self.net(output)