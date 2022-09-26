
#from torch._C import int64
from influence_function import cal_influence_function_single
import torch
import numpy as np 
import dill
import argparse
import os
from get_vectors import getVectors
from vmask_encoder import MASK_LSTM
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


parser = argparse.ArgumentParser(description='_')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-beta', type=float, default=1, help='beta')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='original number of embedding dimension')
parser.add_argument('-lstm-hidden-dim', type=int, default=100, help='number of hidden dimension')
parser.add_argument('-lstm-hidden-layer', type=int, default=1, help='number of hidden layers')
parser.add_argument('-mask-hidden-dim', type=int, default=300, help='number of hidden dimension')
parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=250, help='max sentence length')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--mode', type=str, default='static', help='available models: static, non-static')
parser.add_argument('--save',type=str,default='where_is_model.pt',help='path to save model')
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
args = parser.parse_args()

if args.gpu>-1:
    args.device="cuda"
else:
    args.device="cpu"

dir_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

data=dill.load(open("imdb_data.pkl","rb"))
train_text=data.train_text
train_label=data.train_label
dev_text=data.dev_text
dev_label=data.dev_label


train_text=train_text[:800]
train_label=train_label[:800]
dev_text=dev_text[:160]
dev_label=dev_label[:160]


# print(type(train_text[0]),type(train_label[0]))
# print(type(dev_text[0]),type(dev_label[0]))
wordvocab=data.wordvocab

vectors=getVectors(args,wordvocab)

args.embed_num = len(wordvocab)
args.class_num = 2

# random.seed(1111)
# torch.manual_seed(1111)
# np.random.seed(1111)

criterion = nn.CrossEntropyLoss()

class B:
    text=torch.zeros(1,dtype=torch.int64).to(args.device)
    label=torch.zeros(1,dtype=torch.int64).to(args.device)

def batch_from_list(textlist,labellist):
    batch=B()
    batch.text=textlist[0].unsqueeze(0)
    batch.label=torch.Tensor([labellist[0]])
    # print(batch.text)
    # print(batch.label)
    for txt,la in zip(textlist[1:],labellist[1:]):
        # print("txt, la: ", txt.shape, la.shape)
        # print("batch txt la: ", batch.text.shape, batch.label.shape)
        txt = txt.unsqueeze(0)
        la = torch.Tensor([la])
        batch.text = torch.cat((batch.text, txt), dim=0)
        batch.label = torch.cat((batch.label, la), dim=0)
    batch.text=batch.text.to(args.device)
    batch.label=batch.label.to(args.device)
    
    return batch


def try_check(train_label_flipped,train_label,model,train_text,args,dev_text,dev_label,idx_to_check,label):
        train_label_fixed=np.copy(train_label_flipped)
        train_label_fixed[idx_to_check]=train_label[idx_to_check]
        train_label_flipped=np.array(train_label_flipped)
        train_label_fixed=np.array(train_label_fixed)

        check_num=np.sum(train_label_fixed!=train_label_flipped)
        _ , _ = training(model,train_text,train_label_fixed,args)
        del model
        with open(args.save,'rb') as f:
            model=torch.load(f)
        model.to(torch.device(args.device))
        check_loss, check_acc = evaluation(model,dev_text,dev_label)
        print('%20s: fixed %3s labels. Loss %.5f. Accuracy %.3f.' %(
            label, check_num, check_loss, check_acc
        ))


def test_mislabeled_detection_batch(model,train_text,train_label,train_label_flipped,
    dev_text,dev_label,
    loss,loss_influence,num_checks,
    args):
    assert num_checks>0
    print('num of checks:',num_checks)
    idx_to_check=np.argsort(loss_influence)[-num_checks:]
    print('idx_to_check of influence loss is:', idx_to_check)
    try_check(train_label_flipped,train_label,model,
    train_text,args,dev_text,dev_label,idx_to_check,'Influence (LOO)')

    idx_to_check=np.argsort(loss)[-num_checks:]
    print('idx_to_check of loss is:', idx_to_check)
    try_check(train_label_flipped,train_label,model,
    train_text,args,dev_text,dev_label,idx_to_check, 'Loss')

    idx_to_check=np.random.choice(num_train_examples,size=num_checks,replace=False)
    print('idx_to_check of random is:', idx_to_check)
    try_check(train_label_flipped,train_label,model,
    train_text,args,dev_text,dev_label,idx_to_check,'Random')


def evaluation(model,data_text,data_label):
    model.eval()
    acc,loss,size=0,0,0
    count=0
    
    for stidx in range(0,len(data_label),args.batch_size):
        count+=1
        batch=batch_from_list(data_text[stidx:stidx+args.batch_size],
                                data_label[stidx:stidx+args.batch_size])
        batch.text=torch.squeeze(batch.text)
        pred=model(batch,'eval')
        batch.label=batch.label.long()
        batch_loss=criterion(pred,batch.label)
        loss+=batch_loss.item()

        _, pred=pred.max(dim=1)
        acc+=(pred==batch.label).sum().float()
        size+=len(pred)

    acc/=size
    loss/=count
    return loss,acc

 

def training(model,data_text,data_label,args):
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    model.train()
    beta=args.beta
    best_val_acc=None
    for epoch in range(1,args.epochs+1):
        model.train()
        lstm_count=0
        trn_lstm_size,trn_lstm_corrects,trn_lstm_loss=0,0,0

        listpack=list(zip(data_text,data_label))
        random.shuffle(listpack)
        data_text_pre,data_label_pre=list(zip(*listpack))[0],np.array(list(zip(*listpack))[1]).astype(np.int64)
        data_text[:],data_label[:]=data_text_pre,torch.from_numpy(data_label_pre)
        #data_text[:],data_label[:]=torch.from_numpy(data_text[:]),torch.from_numpy(data_label[:])
        

        for stidx in range(0,len(data_label),args.batch_size):
            lstm_count+=1
            # print('-'*30)
            # print(data_text[0])
            # print(data_label[0])
            # print('-'*30)
            # print(data_text[1])
            # print(data_label[1])

            # print(type(data_text))
            # print(type(data_label))
            
            batch=batch_from_list(data_text[stidx:stidx+args.batch_size],
                                    data_label[stidx:stidx+args.batch_size])
            #print(batch.text.shape)
            batch.text=torch.squeeze(batch.text)
            #print(batch.text.shape)
            with torch.backends.cudnn.flags(enabled=False):
                pred=model(batch,'train')
            optimizer.zero_grad()
            batch.label=batch.label.long()
            model_loss=criterion(pred,batch.label)
            batch_loss=model_loss+beta*model.infor_loss
            trn_lstm_loss+=batch_loss.item()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=args.clip)
            optimizer.step()

            _,pred=pred.max(dim=1)
            trn_lstm_corrects+=(pred==batch.label).sum().float()
            trn_lstm_size+=len(pred)
        dev_lstm_loss,dev_lstm_acc=evaluation(model,dev_text.copy(),dev_label.copy())
        if not best_val_acc or dev_lstm_acc>best_val_acc:
            with open(args.save,'wb') as f:
                torch.save(model,f)
            best_val_acc=dev_lstm_acc
        train_lstm_acc=trn_lstm_corrects/trn_lstm_size
        train_lstm_loss=trn_lstm_loss/lstm_count

        if epoch%10==0:
            if beta>0.01:  
                beta-=0.099
    return train_lstm_acc,train_lstm_loss

np.random.seed(42)

num_flip_vals=5
num_check_vals=5
num_random_seeds=40

num_train_examples=len(train_label)

dims=(num_flip_vals,num_check_vals,num_random_seeds,3)
fixed_influence_loo_results=np.zeros(dims)
fixed_loss_results=np.zeros(dims)

model = MASK_LSTM(args,vectors)
model.to(torch.device(args.device))


for flips_idx in range(num_flip_vals):
    for random_seed_idx in range(num_random_seeds):

        random_seed=flips_idx*(num_random_seeds*3)+(random_seed_idx*2)
        np.random.seed(random_seed)

        num_flips=int(num_train_examples/20)*(flips_idx+1)
        idx_to_flip=np.random.choice(num_train_examples,size=num_flips,replace=False)
        train_label_flipped=np.copy(train_label)
        train_label_flipped=train_label_flipped.astype(np.int64) 

        train_label_array=np.empty(len(train_label),dtype=np.int64) #train_label的复制品
        train_label_array=np.array(train_label,dtype=np.int64)
        

        train_label_flipped[idx_to_flip]=1-train_label_array[idx_to_flip] 

        print('idx_to_flip is: ',idx_to_flip)

        #train_label_flipped_array=train_label_flipped_array.astype(np.int64)
        #train_label_array=train_label_array.astype(np.int64)
        #train_label_flipped=torch.from_numpy(train_label_flipped_array)
        #train_label=torch.from_numpy(train_label_array)
        #print('******************************')
        # print(type(train_text),type(train_label_flipped))
        # print('******************************')
        
        # for i in range(len(train_label_flipped)):
        #     train_text[i],train_text[0]=train_text[0],train_text[i]
        #     train_label_flipped[i],train_label_flipped[0]=train_label_flipped[0],train_label_flipped[i]
        #     train_label_flipped=np.array(train_label_flipped)
        #     _,loss=training(model,train_text[1:],train_label_flipped[1:],args)
        #     train_loss.append(loss)
        #     del loss
        model1=model
        
        train_loss=[]
        train_label_flipped=list(train_label_flipped)
        _,loss=training(model,train_text,train_label_flipped,args)
        del model
        with open(args.save,'rb') as f:
            model=torch.load(f)
        for tex,lab in zip(train_text,train_label_flipped):
            batch=B()
            batch.text=tex.cuda()
            lab=lab.unsqueeze(0).cuda()
            div_loss=criterion(model(batch,'eval'),lab)
            div_loss=div_loss.cpu()
            div_loss=div_loss.detach().numpy()
            train_loss.append(abs(div_loss))

        model=model1
        param=list(model.parameters())
        train_label_flipped=np.array(train_label_flipped)
        train_label_flipped=torch.from_numpy(train_label_flipped)
        train_label=torch.from_numpy(train_label_array)
        #recursion_depth=train_label_flipped.shape[0]
        train_loo_influences, _, _, _ = cal_influence_function_single(model,train_text,train_label_flipped, dev_text,dev_label,0,gpu=0)

        print('Start testing...')
        for checks_idx in range(num_check_vals):
            np.random.seed(random_seed+1)
            num_checks=int(num_train_examples/20)*(checks_idx+1)

            
            test_mislabeled_detection_batch( 
                    model,
                    train_text,train_label,
                    train_label_flipped,
                    dev_text,dev_label,
                    train_loss,train_loo_influences,
                    num_checks,args
                )

            print('I have just finished a round of calculations')