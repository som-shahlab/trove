import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import pdb

class SequenceLabelingClassifier(nn.Module):
    """

    """
    def __init__(self,
                 num_classes,
                 bert_model='bert-base-cased',
                 top_rnns=False,
                 device='cpu',
                 finetuning=False):
        """

        :param num_classes:
        :param bert_model:
        :param top_rnns:
        :param device:
        :param finetuning:
        """
        super().__init__()
        # load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model) if type(bert_model) is str else bert_model

        # use LSTM as top layer (matches original BERT paper's "feature-based approach")
        self.top_rnns = top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True,
                               num_layers=2,
                               input_size=768,
                               hidden_size=768 // 2,
                               batch_first=True)
        self.fc = nn.Linear(768, num_classes)

        self.device = device
        self.finetuning = finetuning

        print('top_rnns', top_rnns)
        print('num_classes', num_classes)

    def forward(self, x, y, ):
        """

        :param x:
        :param y:
        :return:
        """
        x = x.to(self.device).long()
        y = y.to(self.device)

        if self.training and self.finetuning:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


class BaseBERT(nn.Module):
    """

    """
    def __init__(self,
                 bert_model='bert-base-cased',
                 top_rnns=False,
                 device='cpu',
                 finetuning=False):
        """

        :param num_classes:
        :param bert_model:
        :param top_rnns:
        :param device:
        :param finetuning:
        """
        super().__init__()
        # load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model) if type(bert_model) is str else bert_model

        # use LSTM as top layer (matches original BERT paper's "feature-based approach")
        self.top_rnns = top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True,
                               num_layers=2,
                               input_size=768,
                               hidden_size=768 // 2,
                               batch_first=True)

        self.device = device
        self.finetuning = finetuning

    def forward(self, x):
        """

        :param x:
        :param y:
        :return:
        """
        x = x.long()

        if self.training and self.finetuning:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)

        #logits = self.fc(enc)
        #y_hat = logits.argmax(-1)
        return enc


class RNNHead(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.rnn = nn.LSTM(bidirectional=True,
                       num_layers=2,
                       input_size=768,
                       hidden_size=768 // 2,
                       batch_first=True)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        enc, _ = self.rnn(x)
        return self.fc(enc)

class EncoderTransformer(nn.Module):
    """

    """
    def __init__(self,
                 num_classes,
                 top_rnns=True,
                 device='cpu',
                 finetuning=False,
                 pretrained_embeddings=None):
        """

        :param num_classes:
        :param bert_model:
        :param top_rnns:
        :param device:
        :param finetuning:
        """
        super().__init__()

        # embeddings layer
        # encode text

        if pretrained_embeddings is not None:

            self.text_field_embedder = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            vocab_size, embedding_dim = pretrained_embeddings.shape
        else:
            raise ValueError('Pre-trained embeddings must be specified')

        # transformer layer
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                            nhead=5,
                                                            dim_feedforward=500,
                                                            dropout=0.16)

        # use LSTM as top layer (matches original BERT paper's "feature-based approach")
        self.top_rnns = top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True,
                               num_layers=2,
                               input_size=embedding_dim,
                               hidden_size=550 // 2,
                               batch_first=True)

        self.fc = nn.Linear(550, num_classes)

        self.device = device

        print('top_rnns', top_rnns)
        print('num_classes', num_classes)

    def forward(self, x, y, ):
        """

        :param x:
        :param y:
        :return:
        """
        x = x.to(self.device).long()
        y = y.to(self.device)

        embedded_text_input = self.text_field_embedder(x)
        encoded_text = self.transformer(embedded_text_input)

        if self.top_rnns:
            enc, _ = self.rnn(encoded_text)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
