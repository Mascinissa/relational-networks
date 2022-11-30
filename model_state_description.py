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
#         x = self.relu(self.fc3(x))
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
#         print(x.shape)
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
#         pred = output.data.max(1)[1]
#         correct = pred.eq(label.data).cpu().sum()
#         accuracy = correct * 100. / len(label)
#         return accuracy
        
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
#         self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

#         self.coord_oi = torch.FloatTensor(args.batch_size, 2)
#         self.coord_oj = torch.FloatTensor(args.batch_size, 2)
#         if args.cuda:
#             self.coord_oi = self.coord_oi.cuda()
#             self.coord_oj = self.coord_oj.cuda()
#         self.coord_oi = Variable(self.coord_oi)
#         self.coord_oj = Variable(self.coord_oj)

#         # prepare coord tensor
#         def cvt_coord(i):
#             return [(i/5-2)/2., (i%5-2)/2.]
        
#         self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
#         if args.cuda:
#             self.coord_tensor = self.coord_tensor.cuda()
#         self.coord_tensor = Variable(self.coord_tensor)
#         np_coord_tensor = np.zeros((args.batch_size, 25, 2))
#         for i in range(25):
#             np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
#         self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, tabs, qst):
        x = self.fc_in(tabs) ## x = (64 x 24 x 5 x 5) ## x= (64x6x100)
        
        """g"""
        mb = x.size()[0]
#         n_channels = x.size()[1]
#         d = x.size()[2]
#         # x_flat = (64 x 25 x 24)
#         x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        x_flat = x
        
        # add coordinates
#         x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,6,1)
        qst = torch.unsqueeze(qst, 2)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,6,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,6,1) # (64x25x25x26+11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        
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
