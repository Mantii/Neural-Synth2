import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        stride=2
        ker=2
        channels=128
        self.convs=nn.ModuleList([
            nn.Conv1d(1,channels,1,1,0,dilation=1),        nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.AvgPool1d(61)
        ])
        self.meanL1=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,118),nn.BatchNorm1d(118),nn.ReLU()
        )
        self.sigmaL1=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,118),nn.BatchNorm1d(118),nn.ReLU()
        )
        self.meanL2=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,127),nn.BatchNorm1d(127),nn.ReLU()
        )
        self.sigmaL2=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,127),nn.BatchNorm1d(127),nn.ReLU()
        )
        self.meanL3=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,118),nn.BatchNorm1d(118),nn.ReLU()
        )
        self.sigmaL3=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,118),nn.BatchNorm1d(118),nn.ReLU()
        )
        self.pitchDec=nn.Sequential(
            nn.Linear(128,125),nn.BatchNorm1d(125),nn.ReLU(),
            nn.Linear(125,125),nn.BatchNorm1d(125),nn.ReLU()
        )
        self.qualDec=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,250),nn.BatchNorm1d(250),nn.ReLU(),
            nn.Linear(250,250),nn.BatchNorm1d(250),nn.ReLU()
 
        )

        self.classDec=nn.Sequential(
            nn.Linear(128,125),nn.BatchNorm1d(125),nn.ReLU(),
            nn.Linear(125,125),nn.BatchNorm1d(125),nn.ReLU()

        )
        self.jointDec=nn.Sequential(
            nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
            nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
            )
        self.anoDec=[
            nn.ModuleList([nn.Linear(500,1000).cuda(),nn.ReLU()]) for i in range(64)
            ]

        # self.anoDec=[
        #     nn.ModuleList([nn.Linear(500,500).cuda(),nn.ReLU(),
        #     nn.Linear(500,1000).cuda(),nn.ReLU(),nn.Linear(1000,1000).cuda(),nn.ReLU()]) for i in range(64)
        #     ]

        # self.up=nn.Upsample(scale_factor=2)
        # self.UpDec=nn.Sequential(
        #     nn.Linear(1024,1024),nn.ReLU(),
        #     nn.Linear(1024,1024)
        # )


    def sample_latent(self,x):
        #qualities
        mean1=self.meanL1(x)
        sigma1=torch.sqrt(torch.exp(self.sigmaL1(x)))
        #pitch
        mean2=self.meanL2(x)
        sigma2=torch.sqrt(torch.exp(self.sigmaL2(x)))
        #class
        mean3=self.meanL3(x)
        sigma3=torch.sqrt(torch.exp(self.sigmaL3(x)))        
        self.mean=[mean1,mean2,mean3]
        self.sigma=[sigma1,sigma2,sigma3]
        eps1 = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma1.size())
        eps2 = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma2.size())
        eps3 = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma3.size())
        z1=mean1+sigma1*Variable(eps1,requires_grad=False).cuda()
        z2=mean2+sigma2*Variable(eps2,requires_grad=False).cuda()
        z3=mean3+sigma3*Variable(eps3,requires_grad=False).cuda()
        return z1,z2,z3
    
    def forward(self,data):
        x=data['audio'].cuda()
        for L in self.convs:
            x=L(x)
        x=x.view(x.shape[0],128)
        z1,z2,z3=self.sample_latent(x)

        fp=data['pitch'].view(x.shape[0],1).type('torch.FloatTensor')


        z2=self.pitchDec(torch.cat((z2,fp.cuda()),dim=1))
        z1=self.qualDec(torch.cat((z1,torch.FloatTensor(data['qualities']).cuda()),dim=1))
        z3=self.classDec(torch.cat((z3,torch.FloatTensor(data['instrument_family']).cuda()),dim=1))

        z2=torch.cat((z2,z3),dim=1)
        z=torch.cat((z1,z2),dim=1)

        x=self.jointDec(z)
        for count,lays in enumerate(self.anoDec):
            temp=x.cuda()
            for func in lays:
                temp=func(temp)
            if count==0:
                recon=temp
            else:
                recon=torch.cat((recon,temp),dim=1)


        return torch.sigmoid(recon.view(x.shape[0],1,64000))






















