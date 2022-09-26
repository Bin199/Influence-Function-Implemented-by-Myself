import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


SMALL = 1e-08


class VMASK(nn.Module):
	def __init__(self, args):
		super(VMASK, self).__init__()

		self.device = args.device
		self.mask_hidden_dim = args.mask_hidden_dim
		self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.leaky_relu}
		self.activation = self.activations[args.activation]
		self.embed_dim = args.embed_dim
		self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)
		self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)

		#print('embed_dim={}'.format(self.embed_dim))

	def forward_sent_batch(self, embeds):

		temps = self.activation(self.linear_layer(embeds))
		p = self.hidden2p(temps)  # seqlen, bsz, dim
		return p #p表示概率

	def forward(self, x, p, flag):
		if flag == 'train':
			r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2] #hard – 如果 True, 返回的样本将会离散为 one-hot 向量, 但将会是可微分的
			
			x_prime = r * x #..就像是在自动求导的soft样本一样
			return x_prime #返回从 Gumbel-Softmax 分布采样的 tensor, 形状为 batch_size x num_features
		else:              #如果 hard=True, 返回值是 one-hot 编码, 否则, 它们就是特征和为1的概率分布
			probs = F.softmax(p,dim=2)[:,:,1:2] #select the probs of being 1  #dim:A dimension along which softmax will be computed. Default: -1.
		
			x_prime = probs * x
			return x_prime

	def get_statistics_batch(self, embeds):
		p = self.forward_sent_batch(embeds)
		return p


class LSTM(nn.Module):
	def __init__(self, args, vectors):
		super(LSTM, self).__init__()

		self.args = args

		self.embed = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1) #输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，

		# initialize word embedding with pretrained word2vec
		self.embed.weight.data.copy_(torch.from_numpy(vectors)) #生成返回的tensor会和ndarry共享数据，任何对tensor的操作都会影响到ndarry,反之亦然

		# fix embedding
		if args.mode == 'static':
			self.embed.weight.requires_grad = False
		else:
			self.embed.weight.requires_grad = True

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.embed.weight.data[0], -0.05, 0.05) #Fills the input Tensor with values drawn from the uniform distribution \mathcal{U}(a, b)U(a,b) .

		# <pad> vector is initialized as zero padding
		nn.init.constant_(self.embed.weight.data[1], 0)

		# lstm
		self.lstm = nn.LSTM(args.embed_dim, args.lstm_hidden_dim, num_layers=args.lstm_hidden_layer)#setting num_layers=2 would mean stacking two LSTMs 
		                                                                                          #together to form a stacked LSTM, 
																								  # with the second LSTM taking in outputs of the first LSTM and computing the final results

		# initial weight
		init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(6.0)) #gain代表增益大小
		init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(6.0))

		# linear
		self.hidden2label = nn.Linear(args.lstm_hidden_dim, args.class_num)
		# dropout
		self.dropout = nn.Dropout(args.dropout)
		self.dropout_embed = nn.Dropout(args.dropout)

	def forward(self, x):
		# lstm
		lstm_out, _ = self.lstm(x)
	
		lstm_out = torch.transpose(lstm_out, 0, 1)#dim0 (int) – the first dimension to be transposed
		
		lstm_out = torch.transpose(lstm_out, 1, 2)#dim1 (int) – the second dimension to be transposed
		
		# pooling
		lstm_out = torch.tanh(lstm_out)
		
		lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2) #sequeeze(2)在2的位置去掉了一个维度
		
		lstm_out = torch.tanh(lstm_out)
		
		lstm_out = F.dropout(lstm_out, p=self.args.dropout, training=self.training)#p代表置零的概率，tensorflow相反
		
		# linear                                                                    #training – apply dropout if is True. Default: True
		logit = self.hidden2label(lstm_out)
		
		out = F.softmax(logit, 1)
		
		return out


class MASK_LSTM(nn.Module):

	def __init__(self, args, vectors):
		super(MASK_LSTM, self).__init__()
		self.args = args
		self.embed_dim = args.embed_dim
		self.device = args.device
		#self.sample_size = args.sample_size
		self.max_sent_len = args.max_sent_len

		self.vmask = VMASK(args)
		self.lstmmodel = LSTM(args, vectors)

	def forward(self, batch, flag):
		# embedding
		x = batch.text.t()
		embed = self.lstmmodel.embed(x)
		embed = F.dropout(embed, p=self.args.dropout, training=self.training)
		x = embed.view(len(x), embed.size(1), -1)  # seqlen, bsz, embed-dim
		# MASK
		p = self.vmask.get_statistics_batch(x)
		x_prime = self.vmask(x, p, flag)
		output = self.lstmmodel(x_prime)

		# self.infor_loss = F.softmax(p,dim=2)[:,:,1:2].mean()
		probs_pos = F.softmax(p,dim=2)[:,:,1]
		probs_neg = F.softmax(p,dim=2)[:,:,0]
		self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))

		return output
