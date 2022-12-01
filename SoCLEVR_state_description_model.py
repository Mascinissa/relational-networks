import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class FCInputModel(nn.Module):
    def __init__(self):
        super(FCInputModel, self).__init__()
        
        self.fc1 = nn.Linear(10, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)
        self.relu = nn.ReLU()

        
    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

  

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_tabs, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_tabs, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()

        
    def test_(self, input_tabs, input_qst, label):
        output = self(input_tabs, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.fc_in = FCInputModel()
        
        ##(number of filters per object+coordinate of object)*2+question vector
        # 2 obejcts of 100 + qst 11
        self.g_fc1 = nn.Linear((100)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, tabs, qst):
        x = self.fc_in(tabs) 
        
        """g"""
        mb = x.size()[0]
        x_flat = x
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,6,1)
        qst = torch.unsqueeze(qst, 2)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) 
        x_i = x_i.repeat(1,6,1,1) 
        x_j = torch.unsqueeze(x_flat,2) 
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,6,1) 
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) 
        
        # reshape for passing through network
        x_ = x_full.view(mb*6*6,(100)*2+11)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,6*6,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)
