import processingData
import hyperparameter
import torch
import train_crf
import model_lstm
import model_lstm_crf
from torch.autograd import Variable

if __name__ =="__main__":
    param = hyperparameter.Hypermater()
    train_words, train_labels = processingData.read_cropus(param.train_path)
    param.train_size = len(train_words)
    # print(param.train_size)
    dev_words, dev_labels = processingData.read_cropus(param.dev_path)
    param.dev_size = len(dev_words)
    test_words, test_labels = processingData.read_cropus(param.test_path)
    param.test_size = len(test_words)
    word2id, id2word = processingData.create_vocabDict(train_words)
    print(word2id)
    label2id, id2label = processingData.create_labelDict(train_labels)
    print(label2id)

    param.words_dict = word2id
    param.labels_dict = label2id

    train_words_index, word2id, id2word = processingData.sentence_to_index(train_words, word2id, id2word)
    # print(len(train_words_index))
    train_labels_index, label2id, id2label  = processingData.label_to_index(train_labels, label2id, id2label)
    param.label_num = len(id2label)
    param.word_num = len(id2word)
    # print(id2label)
    # train_labels_index = processingData.sentence_to_index(train_labels, label2id, id2label)

    # print(train_labels_index)
    # print(len(train_labels_index))
    # dev_words_index = processingData.sentence_to_index(dev_words, word2id, id2word)
    # dev_labels_index = processingData.sentence_to_index(dev_labels, label2id, id2label)
    # test_words_index = processingData.sentence_to_index(test_words,word2id, id2word)
    # test_labels_index = processingData.sentence_to_index(test_labels,label2id, id2label)

    # train_words_var = Variable(torch.LongTensor(train_words_index))
    # train_labels_var = Variable(torch.LongTensor(train_labels_index))
    # print(train_labels_var)
    # dev_words_var = Variable(torch.LongTensor(dev_words_index))
    # dev_labels_var = Variable(torch.LongTensor(dev_labels_index))
    # test_words_var = Variable(torch.LongTensor(test_words_index))
    # test_labels_var = Variable(torch.LongTensor(test_labels_index))

    if param.use_crf:
        bilstm = model_lstm_crf.LSTM(param)
        print(bilstm)
        if param.use_cuda:
            bilstm = bilstm.cuda()
        if param.proportional:
            train_crf.train(train_labels_index, train_words_index, label2id,  id2label, bilstm, param)
            # train.train(train_labels_var, train_words_var,dev_labels_var,
            #             dev_words_var, test_labels_var, test_words_var, label2id, id2label, bilstm, param)
    else:
        bilstm = model_lstm.LSTM(param)
        if param.use_cuda:
            bilstm = bilstm.cuda()






