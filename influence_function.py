from test_cal_influence import cal_grad, inverse_hvp,cal_loss
import torchvision
import torch
import numpy as np
import torch.nn as nn

if torch.cuda.is_available():
    device=torch.device('cuda')


def cal_s_test_single(model,test_text,test_label,train_text,train_label,gpu=0,
                    damp=0.01,scale=25): #test_text,test_label是单个数据
    # inverse_hvp_list=[]
    
    inverse_hvp_vec=inverse_hvp(test_text,test_label,model,train_text,train_label,
                                    gpu=gpu,damp=damp,scale=scale,
                                    )
    #inverse_hvp_vec=inverse_hvp_list[0]
    # for i in range(1,r):
    #     inverse_hvp_vec+=inverse_hvp_list[i]
    # inverse_hvp_vec=[i/r for i in inverse_hvp_vec]
    return inverse_hvp_vec

def cal_influence_function_single(model,train_text,train_label,test_text,test_label,test_id,gpu,inverse_hvp_vec=None):
    
    if not inverse_hvp_vec:
        test_tex,test_lab=test_text[test_id],test_label[test_id]
        # print(type(test_tex))
        test_tex=test_tex.unsqueeze(0)
        # print(type(test_tex))
        # print(test_tex.shape,test_lab.shape)
        inverse_hvp_vec=cal_s_test_single(model,test_tex,test_lab,train_text,train_label,gpu=gpu)
    print('Start calculating influence value...')
    train_dataset_size=len(train_label)
    influences=[]
    for i in range(train_dataset_size):
        train_tex,train_lab=train_text[i],train_label[i]
        train_tex=train_tex.unsqueeze(0)
        train_lab=train_lab.unsqueeze(0)
        grad_lab_tex=cal_grad(train_tex,train_lab,model,gpu=gpu)
        tmp_influence=-sum([
            torch.sum(k*j).data for k,j in zip(grad_lab_tex,inverse_hvp_vec)
        ])/train_dataset_size
        influences.append(tmp_influence)
    influences1=[]
    for elem in influences:
        influences1.append(elem.cpu().numpy())
    harmful=np.argsort(influences1)
    helpful=harmful[::-1]
    return influences1,harmful.tolist(),helpful.tolist(),test_id



# model = torchvision.models.resnet18(pretrained=True).to(device)
# train_text = torch.rand(1, 3, 64, 64).to(device)
# train_labels = torch.rand(1, 1000).to(device)

# test_text=torch.rand(2, 3, 64, 64).to(device)
# test_labels=torch.rand(2, 1000).to(device)

# train_labels=train_labels.to(device=device,dtype=torch.float)
# test_labels=test_labels.to(device=device,dtype=torch.float)

# model=nn.LSTM(10,20,2).to(device)
# train_input = torch.randn(5, 3, 10).to(device)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = model(input, (h0, c0))
# train_output=output.to(device)

# test_input=train_input
# test_output=train_output

# recursion_depth=train_output.shape[0]
# res=cal_influence_function_single(model,train_input,train_output,test_input,test_output,1,gpu=0,recursion_depth=recursion_depth,r=1)



# params=model.parameters()
# recursion_depth=train_labels.shape[0]

# res=cal_influence_function_single(model,train_text,train_labels,test_text,test_labels,1,gpu=0,recursion_depth=recursion_depth,r=1)
