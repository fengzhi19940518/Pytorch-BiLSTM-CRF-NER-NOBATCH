import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import processingData
import random
torch.manual_seed(223)
random.seed(223)

def train(train_labels_index, train_words_index,label2id, id2label, model, params):
    print("executing train function...")
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    steps = 0
    for epoch in range(1, params.epochs + 1):
        train_len = train_labels_var.size(0)
        perm_list = torch.randperm(train_len)
        train_labels_var = train_labels_var[perm_list]
        train_words_var = train_words_var[perm_list]
        train_words_iter, train_labels_iter = processingData.create_beaches(train_words_var, train_labels_var, params.batch_size)
        for index in range(len(train_labels_iter)):
            model.zero_grad()
            if params.use_cuda:
                train_words_iter[index] = train_words_iter[index].cuda()
                train_labels_iter[index] = train_labels_iter[index].cuda()
            optimizer.zero_grad()
            logit = model.forward(train_words_iter[index])
            # batch_size = train_labels_iter[index].size(0)
            # loss = F.cross_entropy(logit, train_labels_iter[index].squeeze(1))
            loss = model.crf.cal_loss(logit, train_labels_iter[index])
            loss.backward()
            optimizer.step()
            steps += 1

            evaluating_standard = []
            for i in range(3):
                evaluating_params = []
                for j in range(len(id2label)):
                    evaluating_params.append(0)
                evaluating_standard.append(evaluating_params)
            correct_num = 0
            if steps % params.train_print_acc == 0:
                predict = torch.max(logit, 1)[1].data.tolist()
                label_iter_index_list = train_labels_iter[index].squeeze(1).data.tolist()









