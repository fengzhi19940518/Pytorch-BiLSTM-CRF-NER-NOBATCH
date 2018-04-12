import numpy as np
import torch
from torch.autograd import Variable


class Alphabet():
    def __init__(self):
        self.string2id = {}
        self.id2string = []

def read_cropus(path):
    allword_list = []
    alllabel_list = []
    words = []
    labels = []
    sen = ' '
    lab = ' '
    with open(path, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            if line != "\n":
                sentences = line.strip().split(' ')
                if sentences[0] == "token":
                    sen = sen + sentences[1] + ' '
                    lab = lab + sentences[-1] + ' '
            else:
                words.append(sen.strip())
                labels.append(lab.strip())
                sen = ' '
                lab = ' '
    for i in range(len(words)):
        text = words[i].strip().lower()
        label = labels[i].strip()
        allword_list.append(text.split())
        alllabel_list.append(label.split())
    # print(alllabel_list)
    # print(allword_list)
    return allword_list, alllabel_list

def create_vocabDict(vocab):
    print("creating vocab dict...")
    alpha = Alphabet()
    alpha.id2string.append("unk")
    alpha.string2id["unk"] = 0
    for i in range(len(vocab)):
        for w in vocab[i]:
            if w not in alpha.string2id.keys():
                alpha.id2string.append(w)
                alpha.string2id[w] = len(alpha.id2string) - 1
    print("have finished creating vocab dict.")

    return alpha.string2id, alpha.id2string

def create_labelDict(label):
    print("creating label dict...")
    alpha = Alphabet()
    for i in range(len(label)):
        for w in label[i]:
            if w not in alpha.string2id.keys():
                alpha.id2string.append(w)
                alpha.string2id[w] = len(alpha.id2string) - 1
    print("have finished creating label dict.")
    # print("alpha.string2id",alpha.string2id)
    return alpha.string2id, alpha.id2string

def sentence_to_index(data, string2id, id2string):
    print("convert sentence to index")
    data_index = []
    for i in range(len(data)):
        sen_index = []
        for w in data[i]:
            if w not in string2id.keys():
                sen_index.append(string2id["unk"])
            else:
                sen_index.append(string2id[w])
        data_index.append(sen_index)
    # data_index.sort(key = lambda s: len(s), reverse=True)
        # print(data_index)
    # sentence_index = pad(data_index, string2id, id2string)
        # print(data_index)
    print("finishing converting sentence to index.")
    return data_index, string2id, id2string

def label_to_index(data, string2id, id2string):
    print("convert label to index ")
    data_index = []
    for index in range(len(data)):
        sen_index = []
        for w in data[index]:
            # print("string2id[w]",string2id[w])
            sen_index.append(string2id[w])
        data_index.append(sen_index)
    # label_index = pad(data_index, string2id, id2string)
    print("finishing converting label to index. ")
    return data_index, string2id, id2string

def pad(data_index, string2id, id2string, token_pad = '<pad>'):
    max_len = getMax_len(data_index)
    for i in range(len(data_index)):
        sen_index = data_index[i]
        if len(sen_index) <= max_len:
            if token_pad not in string2id.keys():
                id2string.append(token_pad)
                string2id[token_pad] = len(id2string) - 1
            for j in range(max_len - len(sen_index)):
                sen_index.append(string2id[token_pad])
    return data_index

def getMax_len(data_index):
    max_len = 0
    for i in range(len(data_index)):
        if max_len <= len(data_index[i]):
            max_len = len(data_index[i])
    return max_len

def create_beaches(words_var, labels_var, batch_size):
    print("create batches...")
    words_iter = []
    labels_iter = []
    if words_var.size(0) % batch_size == 0:
        batch_num = words_var.size(0) // batch_size
    else:
        batch_num = words_var.size(0) // batch_size + 1

    for i in range(batch_num):
        temp_iter = []
        for j in range(batch_size):
            if (i * batch_size + j + 1) <= len(words_var):
                temp_iter.append(words_var.data[i * batch_size + j].tolist())
        words_iter.append(Variable(torch.LongTensor(temp_iter)))

    for i in range(batch_num):
        temp_iter = []
        for j in range(batch_size):
            if (i * batch_size + j +1) <= len(labels_var):
                temp_iter.append(labels_var.data[i * batch_size + j].tolist())
        labels_iter.append(Variable(torch.LongTensor(temp_iter)))
    print("Complete to create batches.")
    return words_iter, labels_iter

start_label = ['b', 'B']
middle_label = ['m','M']
end_label = ['e', 'E']
single_label = ['s', 'S']
prefix = ['b', 'B','m','M','e', 'E','s', 'S']
def split_span(predict_path_label):
    label_list = []
    pre_word = ''
    id_pre = -1
    count_agent = 0
    count_dse = 0
    count_target = 0
    for index in range(len(predict_path_label)):
        label = predict_path_label[index]
        if len(label) > 1 :
            label_prefix = label.split('-')[0]
            label_word = label.split('-')[1]
            if label_prefix in start_label:
                id_pre = index
                pre_word = label_word
                continue
            elif label_prefix in middle_label:
                if label_word == pre_word:
                    continue
                else:
                    id_pre = -1
                    pre_word = ''
            elif label_prefix in end_label:
                if label_word == pre_word:
                    label_list.append(label_word+'['+str(id_pre)+','+str(index)+']')
                    if label_word =='AGENT':
                        count_agent += 1
                    elif label_word =='DSE':
                        count_dse += 1
                    elif label_word == 'TARGET':
                        count_target += 1
                id_pre = -1
                pre_word = ''
            elif label_prefix in single_label:
                label_list.append(label_word + '[' + str(index)+ ',' + str(index) + ']')
                if label_word == 'AGENT':
                    count_agent +=1
                elif label_word =='DSE':
                    count_dse += 1
                elif label_word == 'TARGET':
                    count_target += 1
    predict_count = (count_agent, count_dse, count_target)

    return label_list , predict_count

def getFscore(predict_num, gold_num, correct_num,):
        if predict_num == 0:
            precision = 0
        else:
            precision = correct_num / predict_num

        if gold_num == 0:
            recall = 0
        else:
            recall = correct_num / gold_num

        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precision * recall) / (precision + recall)

        return precision, recall, fscore

