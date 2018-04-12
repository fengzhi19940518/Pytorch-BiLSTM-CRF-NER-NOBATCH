import torch
import torch.nn as nn
from torch.autograd import  Variable
a =torch.randn(2,3)
# print(len(a))
# print(a)
#
# trans = nn.Parameter(torch.zeros(5,5))
# print(trans)

score = Variable(torch.Tensor(5,5))
# print(torch.max(score, 1))
# print(score)
# print(score.view(1, -1))
for i in range(len(score)):
    print(score[i].data[0])
# back = [2,1,3,4,7,5]

# for i in reversed(back):
    # print(i)

s = 'hello'
t = []
t.append(s + str('sss'))
# print(t)

b = [1,3,4,2]
c = [2,3,4,5]
# print(list(set(b).intersection(set(c))))
# print(list(set(b).union(set(c))))
# print(list(set(b).difference(set(c))))