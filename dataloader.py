import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable
import dill
import torch.nn as nn


# def cal_influence(param, loss):
#     Hessian=None
#     print(type(loss))
#     print(type(param))
#     loss_=loss
#     loss=torch.Tensor([loss_])
#     loss=Variable(loss,requires_grad=True)
#     grads=autograd.grad(loss,param,create_graph=True)
#     param_test=param.copy()
#     loss_test=loss.copy()
#     grads_test=autograd.grad(loss_test, param_test, create_graph=True)

#     grads_flatten=torch.cat([g.reshape(-1) for g in grads if g is not None])
#     grads_flatten_test=torch.cat([g.reshape(-1) for g in grads_test if g is not None])

#     for grad in  grads_flatten:
#         second_grads=autograd.grad(grad,param,create_graph=True)
#         second_grads_flatten=torch.cat([g.reshape(-1) for g in second_grads if g is not None])
#         second_grads_flatten=second_grads_flatten.unsequeeze(0)

#         if Hessian is None:
#             Hessian=second_grads_flatten
#         else:
#             Hessian=torch.cat((Hessian,second_grads_flatten),dim=0)
    
#     ones_diag=torch.diag(torch.ones(len(Hessian))).cuda()
#     lambda_diag=ones_diag*0.01
#     Hessian=Hessian+lambda_diag

#     H_inverse=torch.inverse(Hessian)
#     influence=grads_flatten_test.t()@(H_inverse@grads_flatten)
#     return influence

data=dill.load(open("imdb_data.pkl","rb"))
train_text=data.train_text
train_label=data.train_label
dev_text=data.dev_text
dev_label=data.dev_label


train_text=train_text[:80]
train_label=train_label[:80]
dev_text=dev_text[:40]
dev_label=dev_label[:40]

for i in range(len(train_label)):
    print(train_text[i])
    print('\n')
    print(train_label[i])