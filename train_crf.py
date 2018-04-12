import torch
from torch.autograd import Variable
import processingData
import random
torch.manual_seed(223)
random.seed(223)

def train(train_labels_index, train_words_index, label2id,  id2label, model, params):
    print("executing train function...")
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    for epoch in range(1, params.epochs + 1):
        steps = 0
        correct_num = 0
        predict_num = 0
        real_num = 0
        #p_agent 、p_dse、p_target、r_agent 、r_dse、r_target、c_agent、c_dse 、c_target
        evaluating_standard = []
        for i in range(3):
            evaluating_params = []
            for j in range(3):
                evaluating_params.append(0)
            evaluating_standard.append(evaluating_params)

        for index in range(params.train_size):
            train_labels_var = Variable(torch.LongTensor(train_labels_index[index]))
            train_words_var = Variable(torch.LongTensor(train_words_index[index]))
            # print(train_labels_var)
            if params.use_cuda:
                train_labels_var = train_labels_var.cuda()
                train_words_var = train_words_var.cuda()

            model.zero_grad()
            optimizer.zero_grad()
            logit = model.forward(train_words_var)
            # print("logit",logit)
            # print("train_var",train_labels_var)
            loss = model.crf.cal_loss(logit, train_labels_var)
            loss.backward()
            optimizer.step()

            predict_path = torch.max(logit, 1)[1].data
            # print(predict_path)
            steps += 1
            sen_len = logit.size(0)
            sen_label_best = []
            for j in range(sen_len):
                label_name = id2label[predict_path[j]]
                sen_label_best.append(label_name)
            # print(sen_label_best)
            sen_label_group_best, predict_count = processingData.split_span(sen_label_best)
            predict_agent = predict_count[0]
            predict_dse = predict_count[1]
            predict_target = predict_count[2]
            predict_num += len(sen_label_group_best)
            evaluating_standard[0][0] += predict_agent
            evaluating_standard[0][1] += predict_dse
            evaluating_standard[0][2] += predict_target

            sen_label = []
            for j in range(sen_len):
                label_name = id2label[train_labels_var[j].data.tolist()[0]]
                # print(label_name)
                sen_label.append(label_name)
            sen_label_group, real_count = processingData.split_span(sen_label)
            real_num += len(sen_label_group)
            evaluating_standard[1][0] += real_count[0]
            evaluating_standard[1][1] += real_count[1]
            evaluating_standard[1][2] += real_count[2]

            for i in range(len(sen_label_group_best)):
                if sen_label_group != []:
                    gold_len = len(sen_label_group)
                    gold_count = [0]* gold_len
                    s = sen_label_group_best[i]
                    s_word = s.split('[')[0]
                    # print(s_word)
                    s_span = s.split('[')[1].split(']')[0].split(',')
                    # print(s_span)
                    s_set = set(range(int(s_span[0]),int(s_span[1])+1))
                    # print(s_set)
                    for id, e in enumerate(sen_label_group):
                        if gold_count[id] == 0:
                            e_word = e.split('[')[0]
                            e_span = e.split('[')[1].split(']')[0].split(',')
                            e_set = set(range(int(e_span[0]),int(e_span[1])+1))
                            if s_word == e_word:
                                if len(s_set.intersection(e_set)):
                                    propor_score = float(len(s_set.intersection(e_set))) / float(len(e_set))
                                    correct_num += propor_score
                                    if s_word == 'AGENT':
                                        evaluating_standard[2][0] += propor_score
                                    elif s_word == 'DSE':
                                        evaluating_standard[2][1] += propor_score
                                    elif s_word == 'TARGET':
                                        evaluating_standard[2][2] += propor_score
            # print(evaluating_standard[0][0])
            if steps % params.batch_size == 0:
                precision, recall, fscore = processingData.getFscore(predict_num, real_num, correct_num)
                print('\rEpoch{} Sentences{} - loss: {:.6f} precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                        epoch, steps, loss.data[0], (precision * 100), (recall * 100), fscore, correct_num, real_num,
                        params.batch_size))

                precision_agent, recall_agent, fscore_agent = processingData.getFscore(evaluating_standard[0][0],evaluating_standard[1][0],evaluating_standard[2][0])
                print('\r   AGENT - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                    (precision_agent * 100), (recall_agent * 100), fscore_agent, evaluating_standard[2][0], evaluating_standard[1][0], params.batch_size))

                precision_dse, recall_dse, fscore_dse = processingData.getFscore(evaluating_standard[0][1],evaluating_standard[1][1],evaluating_standard[2][1])
                print('\r     DSE - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                    (precision_dse * 100), (recall_dse * 100), fscore_dse, evaluating_standard[2][1],evaluating_standard[1][1], params.batch_size))

                precision_target, recall_target, fscore_target = processingData.getFscore(evaluating_standard[0][2],evaluating_standard[1][2],evaluating_standard[2][2])
                print('\r  TARGET - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})\n'.format(
                    (precision_target * 100), (recall_target * 100), fscore_target, evaluating_standard[2][2], evaluating_standard[1][2], params.batch_size))


                #清零
                # p_agent 、p_dse、p_target、r_agent 、r_dse、r_target、c_agent、c_dse 、c_target、predict_num、correct_num、real_num
                evalValue = []
                for i in range(3):
                    evalParams = []
                    for j in range(4):
                        evalParams.append(0)
                    evalValue.append(evalParams)
                if steps == 10000000:
                    count = params.train_size % params.train_print_acc
                    precision, recall, fscore = processingData.getFscore(evalValue[0][3], evalValue[1][3],evalValue[2][3])
                    print('\rEpoch{} Sentences{} - loss: {:.6f} precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                            epoch, steps, loss.data[0], (precision * 100), (recall * 100), fscore, correct_num, real_num, count))

                    precision_agent, recall_agent, fscore_agent = processingData.getFscore(evalValue[0][0],
                                                                                           evalValue[1][0],
                                                                                           evalValue[2][0])
                    print('\r   AGENT - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                        (precision_agent * 100), (recall_agent * 100), fscore_agent, evalValue[2][0],
                        evalValue[1][0], params.batch_size))

                    precision_dse, recall_dse, fscore_dse = processingData.getFscore(evalValue[0][1],
                                                                                     evalValue[1][1],
                                                                                     evalValue[2][1])
                    print('\r     DSE - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                        (precision_dse * 100), (recall_dse * 100), fscore_dse, evalValue[2][1],
                        evalValue[1][1], params.batch_size))

                    precision_target, recall_target, fscore_target = processingData.getFscore(evalValue[0][2],
                                                                                              evalValue[1][2],
                                                                                              evalValue[2][2])
                    print('\r  TARGET - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})\n'.format(
                        (precision_target * 100), (recall_target * 100), fscore_target, evalValue[2][2],
                        evalValue[1][2], params.batch_size))











