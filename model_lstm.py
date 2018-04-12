import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loadEmbedding import LoadEmbedding
import crf


class LSTM(nn.Module):
    def __init__(self,params):
        super(LSTM, self).__init__()
        self.params = params
        self.hidden_size = params.hidden_size
        self.embedding_word = LoadEmbedding(params.word_num, params.embed_dim)
        self.embedding_word.load_pretrained_embedding(params.embedding_path,
                                                      params.words_dict,
                                                      params.save_words_embedding,
                                                      binary=False)
        self.embedding_label = LoadEmbedding(params.topic_num, params.embed_dim)
        self.embedding_label.load_pretrained_embedding(params.embedding_path,
                                                       params.labels_dict,
                                                       params.save_labels_embedding,
                                                       binary=False)
        self.bilstm =nn.LSTM(params.embed_dim, params.hidden_size, dropout=params.dropout,
                             num_layers=params.num_layers, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(params.hidden_size * 2, params.hidden_size // 2)
        self.linear2 = nn.Linear(params.hidden_size // 2 ,params.label_num)

    def forward(self, words_var):

        x = self.embedding_word(words_var)
        lism_out, _ = self.bilstm(x)
        lism_out =lism_out.squeeze(0)
        # lstm_out = torch.transpose(lism_out, 1, 2)

        tanh_out = F.tanh(lism_out)
        # pooling_out = F.max_pool1d(tanh_out, tanh_out.size(2))
        # squ_out = pooling_out.squeeze(2)
        logit = self.linear1(tanh_out)

        tanh_out2 = F.tanh(logit)
        logit = self.linear2(tanh_out2)

        return logit



