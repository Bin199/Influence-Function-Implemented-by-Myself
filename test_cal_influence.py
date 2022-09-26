from sys import flags
from typing import Counter
import torch
from torch._C import device
from torch.autograd import grad
import torch.nn as nn
from torch.autograd import Variable

criterion= nn.CrossEntropyLoss()

class B:
    text=torch.zeros(1,dtype=torch.int64).to('cuda')
    label=torch.zeros(1,dtype=torch.int64).to('cuda')

def cal_loss(pred,label):
    # pred=torch.nn.functional.log_softmax(pred)
    # loss=torch.nn.functional.nll_loss(
    #     pred,label,weight=None,reduction='mean'
    # )
    #loss=(pred-label).sum()
    loss=criterion(pred,label)
    return loss

def cal_grad(text,label,model,gpu=0,flag='train'):
    model.train()
    #model.zero_grad()
    if gpu>=0:
        text,label=text.cuda(),label.cuda()
    batch=B()
    batch.text=text.squeeze(0)
    batch.label=torch.Tensor([label])
    #batch.text=batch.text.to(device='cuda')
    #batch.label=batch.label.to(device='cuda')
    # print(batch.text.shape)
    with torch.backends.cudnn.flags(enabled=False):
        pred=model(batch,flag)
    #pred=model(text)
    loss=cal_loss(pred,label)
    #model.zero_grad()
    params=[p for p in model.parameters() if p.requires_grad]
    return grad(loss,params,create_graph=True)

def hvp(y,w,v):
    if len(w)!=len(v):
        raise(ValueError("w and v must have the same length."))
    #print(type(y),'*y')
    
    first_grads=grad(y,w,create_graph=True,retain_graph=True) #雅可比矩阵
    # print(first_grads)
    # print(type(first_grads))
    # count=0
    # for _fg in first_grads:
    #     if _fg is not None:
    #         count+=1
    # print(count,len(first_grads))
    # print('*'*30)
    # print(first_grads)
    # print('hello')
    elemwise_products=0
    for grad_elem, v_elem in zip(first_grads,v):
        elemwise_products+=torch.sum(grad_elem*v_elem)
    #print(elemwise_products.shape)
    
    # print(type(elemwise_products),'*elemwise_products')
    return_grads=grad(elemwise_products,w,create_graph=True,allow_unused=True) #hessians矩阵
    
    # elemwise_products=0
    # for grad_elem,v_elem in zip(first_grads,v):
    #     elemwise_products+=torch.sum(grad_elem*v_elem)
    # print(elemwise_products.item())
    # sum1=0
    # return_grads=torch.empty(len(w))
    # for i in range(len(w)):
    #     w[i]=Variable(w[i],requires_grad=True)
    #     sum1+=w[i]
    #     return_grads[i]=grad(sum1,w[i],retain_graph=True)
    #return_grads=grad(first_grads,w,v)


    return return_grads


def inverse_hvp(test_text,test_label,model,train_text,train_label,gpu=0,damp=0.01,scale=25.0): #test_text,test_label是单个数据
    #v=cal_grad(test_text,test_label,model,gpu=gpu,flag='eval')
    v=cal_grad(test_text,test_label,model,gpu=gpu)
    h_estimate=v
    
    
    count=0
    for tex,lab in zip(train_text,train_label):
        count+=1
        #tex=tex.unsqueeze(0)
        lab=lab.unsqueeze(0)
        #print(tex.shape,lab.shape)
        if gpu>=0:
            tex,lab=tex.cuda(),lab.cuda()
        batch=B()
        batch.text=tex
        batch.label=torch.Tensor([lab])
        
        #print(batch.text.shape)
        with torch.backends.cudnn.flags(enabled=False):
            pred=model(batch,flag='train')
        #pred=model(tex)
        #print(type(lab))
        lab=lab.type(torch.long)
        loss=cal_loss(pred,lab)
        #print(type(loss),'*loss')
        params=[p for p in model.parameters() if p.requires_grad]
        #print(type(params[0]),type(h_estimate))

        model.train()
        hv=hvp(loss,params,h_estimate)
        
        
        # for _v, _h_e, _hv in zip(v,h_estimate,hv):
        #     print(_v.shape, _h_e.shape, _hv.shape)

        h_estimate=[_v + (1-damp)*_h_e -_hv/scale
        for _v, _h_e, _hv in zip(v,h_estimate,hv)]
        
        if count>=4:
            break
        
        #清除显存占用
        # model.zero_grad()
        # del batch
        # del loss
        # del hv
        # torch.cuda.empty_cache()
    

    return h_estimate