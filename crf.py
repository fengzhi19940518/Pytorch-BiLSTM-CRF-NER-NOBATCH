import torch
import torch.nn as nn
from torch.autograd import Variable
class CRF(nn.Module):
    def __init__(self,params):
        super(CRF, self).__init__()
        self.params =params
        self.labelSize = params.label_num
        self.T = nn.Parameter(torch.randn(self.labelSize,self.labelSize))

    def sentence_score_gold(self, logit, labels):
        # print(logit)
        # print(labels)
        score = Variable(torch.Tensor([0]))
        if self.params.use_cuda:
            score = score.cuda()

        for idx in range(len(logit)):
            feat = logit[idx]
            if idx == 0:
                score += feat[labels[idx]]
            else:
                # print(self.T[labels[idx].data[0], labels[idx - 1].data[0]])
                score += feat[labels[idx]] + self.T[labels[idx].data[0], labels[idx - 1].data[0]]
        return score

    def to_scalar(self, vec):
        # print("vec",vec.view(-1).data.tolist()[0])
        return vec.view(-1).data.tolist()[0]

    def argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return self.to_scalar(idx)

    def log_sum_exp(self, vec):
        # print(vec)
        max_score = vec[0, self.argmax(vec)]
        # print(max_score)
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        # print("max_score_b",max_score_broadcast)
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def forward_crf(self, logit):
        init_alphas = torch.Tensor(1, self.labelSize).fill_(0)
        # print(init_alphas)
        forward_var = Variable(init_alphas)
        # print("brfore",forward_var)
        if self.params.use_cuda:
            forward_var = forward_var.cuda()

        for idx in range(len(logit)):
            feat = logit[idx]
            alphas_t = []
            for next_tag in range(self.labelSize):
                if idx == 0:
                    alphas_t.append(feat[next_tag].view(1, -1))
                else:
                    emit_socre = feat[next_tag].view(1, -1).expand(1, self.labelSize)
                    # print(emit_socre)
                    trans_score = self.T[next_tag]
                    next_tag_var = forward_var + emit_socre + trans_score
                    alphas_t.append(self.log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
            # print("after", forward_var)
        alpha_score = self.log_sum_exp(forward_var)
        return  alpha_score

    def cal_loss(self, logit, label_index):
        log_score = self.forward_crf(logit)
        gold_score = self.sentence_score_gold(logit, label_index)
        loss = log_score - gold_score
        return loss

    def viterbi_decode(self, logit):
        init_score = torch.Tensor(1,self.labelSize).fill_(0)
        forward_var = Variable(init_score)
        if self.params.use_cuda:
            forward_var = forward_var.cuda()
        back = []
        for idx in range(len(logit)):
            feat = logit[idx]
            bptrs_t = []
            viterbi_var = []
            for next_tag in range(self.labelSize):
                if idx == 0:
                    viterbi_var.append(feat[next_tag].view(1, -1))
                else:
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.labelSize)
                    trans_score = self.T[next_tag]
                    next_tag_var = forward_var + trans_score + emit_score
                    best_label_id = self.argmax(next_tag_var)
                    bptrs_t.append(best_label_id)
                    viterbi_var.append(next_tag_var[0][best_label_id])
            forward_var = (torch.cat(viterbi_var)).view(1, -1)
            if idx > 0:
                back.append(bptrs_t)
        best_label_id = self.argmax(forward_var)
        best_path = [best_label_id]
        path_score = forward_var[0][best_label_id]
        for bptrs_t in reversed(back):
            best_label_id = bptrs_t[best_label_id]
            best_path.append(best_label_id)
        best_path.reverse()

        return path_score, best_path





