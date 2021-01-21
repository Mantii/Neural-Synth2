import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from AdvModel import AutoEncoder
from torch.utils.data import DataLoader
from os import listdir
from torchaudio.transforms import MuLawEncoding
import matplotlib.pyplot as plt
import math
def latent_loss(mean,sd,mul):
    mean2=mean*mean
    sd2=sd*sd
    loss=mul*torch.mean(mean2+sd2-torch.log(sd2)-1)
    fll=float(torch.mean(mean2+sd2-torch.log(sd2)-1))
    return loss,fll
def latent_lossp(mean,sd,mul):
    l1,f1=latent_loss(mean[0],sd[0],mul)
    l2,f2=latent_loss(mean[1],sd[1],mul)
    l3,f3=latent_loss(mean[2],sd[2],mul)
    return l1+l2+l3,f1+f2+f3


num_epochs=100000000
samples=	2
batch_size=	4
check=		True
KL=			False
saveevery=	10
MSE=		True
LR=			1e-3
plot=		False
data=[]
for d in listdir('audio'):
	print('Loading ',d)
	for i,s in enumerate(listdir('audio/'+d)):
		if i==samples or len(data)==1e10:
			break
		info=torch.load('audio/'+d+'/'+s)
		info['pitch']=float(len(info['pitch']))
		info['audio']=MuLawEncoding()(info['audio']).type('torch.FloatTensor')
		info['audio']=(info['audio']-torch.min(info['audio']))/(torch.max(info['audio'])-torch.min(info['audio']))
		data.append(info)
batches=int(len(data)/batch_size)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = data

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
dataclass = MyDataset()
dataset=DataLoader(dataclass,batch_size=batch_size,num_workers=4,drop_last=True)


vae = AutoEncoder().cuda()
crit=nn.BCELoss(reduction='mean')

optimizer = optim.Adam(vae.parameters(),lr=LR)

if check:
    checkpoint = torch.load('save.tar')
    vae.load_state_dict(checkpoint['vae'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("\nLoaded model checkpoint\nEpoch: ",checkpoint['epoch'],'\nLoss:',checkpoint['loss'])
    mul=checkpoint['mul']	
else:
    mul=0
mul=0

# for param_group in optimizer.param_groups:
#         param_group['lr'] = 1e-3

for epoch in range(num_epochs):
	stats={'R':0,'L':0}
	for data in tqdm(dataset,leave=False):
		y=vae(data)

		ll,fll=latent_lossp(vae.mean,vae.sigma,mul)
		loss=crit(y,data['audio'].cuda())+ll
		stats['R']+=float(loss.item()-ll.item())
		stats['L']+=fll
		loss.backward()
		optimizer.step()


	print("{}:\tRL: {}  LL: {}".format(epoch,str(stats['R']/batches)[0:8],str(stats['L']/batches)[0:8]))#total_loss/(samples*10/batch_size)

	if math.isnan(stats['R']) or math.isnan(stats['L']):
		print('\n\nReceived NaN: Breaking...\n\n')
		break

	#  PLOTTING
	if plot:
		plt.plot(y[0].detach().cpu().view(64000).numpy(),'r')
		plt.plot(data['audio'][0].detach().cpu().view(64000).numpy(),'g')
		plt.show()


	#  SAVING
	if epoch%saveevery==0 and epoch!=0:
		torch.save({
		    'vae': vae.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		    'epoch':epoch,#+checkpoint['epoch'],
		    'loss':stats['R']/batches,
		    'mul':mul
		    },'save.tar')
		print("Model Saved.")
		#optimizer = optim.Adam(vae.parameters(),lr=LR)


