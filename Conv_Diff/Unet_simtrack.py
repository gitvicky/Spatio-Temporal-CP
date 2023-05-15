"""
Created on Sat Aug 20, 2022

@author: vgopakum

Demonstration of SimTrack

Surrogate Model for the 1D Convection Diffusion developed using a 1D U-Net Model deployed in a autoregressive framework. 
"""

# %%
configuration = {"Case": 'ConvDiff 1D',
                 "Model": 'U-Net',
                 "Epochs": 500,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'None',
                 "T_in": 20, 
                 "T_out": 60,
                 "Step": 10,
                 "Train Size": 900, 
                 "Test Size": 100
                 }

from simtrack import Simtrack
run = Simtrack()   

from random_word import RandomWords
r = RandomWords()
run_name = r.get_random_word() +'-'+ r.get_random_word()

run.init(run_name,
        configuration, 
        ['PyTorch'],  # Tags
        'Convection Diffusion 1D Surrogate using a U-Net',   # Description
        '/ConvDiff/Mark_2')    # Folder full path

#Upload the code
run.save('Unet_simtrack.py', 'code')



# %%
from collections import OrderedDict

import os 
import numpy as np 
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
from timeit import default_timer
from tqdm import tqdm 
import time  
import operator
from functools import reduce
from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()

class UNet1d(nn.Module):

    def __init__(self, in_channels=20, out_channels=10, init_features=32):
        super(UNet1d, self).__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=3, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

model = UNet1d()
# wandb.run.summary['Number of Params'] = model.count_params()
print("Number of model params : " + str(model.count_params()))
model.to(device)

# %%

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# %%
ntrain = configuration['Train Size']
ntest = configuration['Test Size']
batch_size = configuration['Batch Size']
T_in = configuration['T_in']
step = configuration['Step']
T_out = configuration['T_out']

u = np.load(os.getcwd() + '/Data/ConvDiff_u.npz')['u']
u =  u.astype(np.float32)
u = torch.from_numpy(u)

train_a = u[:ntrain,:T_in,:]
train_u = u[:ntrain,T_in:T_out+T_in,:]

test_a = u[-ntest:,:T_in, :]
test_u = u[-ntest:,T_in:T_out+T_in,:]


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# %% 

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

myloss = torch.nn.MSELoss()
# myloss = LpLoss(size_average=False)


epochs = configuration['Epochs']

for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T_out, step):
            y = yy[:, t:t + step, :]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)

            xx = torch.cat((xx[:, step:, :], im), dim=1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T_out, step):
                y = yy[:, t:t + step, :]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)

                xx = torch.cat((xx[:, step:, :], im), dim=1)
            
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    
    train_loss = train_l2_full / ntrain
    test_loss = test_l2_full / ntest
    
    print('Epochs: %d, Time: %.2f, Train Loss per step: %.3e, Train Loss: %.3e, Test Loss per step: %.3e, Test Loss: %.3e' % (ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_loss, test_l2_step / ntest / (T_out / step), test_loss))
    
    run.log({'Train Loss': train_loss, 
               'Test Loss': test_loss})
    
train_time = time.time() - start_time
# %%
model_loc = os.getcwd() + '/Models/'
torch.save(model.state_dict(), model_loc + run_name + '.pth')
run.save(model_loc  + run_name + '.pth', 'output')


# %%

#Testing 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        for t in range(0, T_out, step):
            y = yy[:, t:t + step, :]
            out = model(xx)
            loss += myloss(out.reshape(1, -1), y.reshape(1, -1))

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), 1)       
                
            xx = torch.cat((xx[:, step:, :], out), dim=1)

        
        # pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
    
MSE_error = (pred_set - test_u).pow(2).mean()
MAE_error = torch.abs(pred_set - test_u).mean()
LP_error = loss / (ntest*T_out/step)

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))

# wandb.run.summary['Training Time'] = train_time
# wandb.run.summary['MSE Test Error'] = MSE_error
# wandb.run.summary['MAE Test Error'] = MAE_error

# %%
idx = np.random.randint(0, ntest) 
print(idx)
u_field_num = test_u[idx].cpu().detach().numpy()
u_field_surr = pred_set[idx].cpu().detach().numpy()

fig = plt.figure(figsize=plt.figaspect(0.25))

ax = fig.add_subplot(1,3,1)
ax.plot(u_field_num[0], color='blue', label='Numerical Solution')
ax.plot(u_field_surr[0], color='red', alpha=0.5, label='Neural Network')
ax.title.set_text('Initial')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()

ax = fig.add_subplot(1,3,2)
ax.plot(u_field_num[int(T_out/2)], color='blue')
ax.plot(u_field_surr[int(T_out/2)], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Middle')

ax = fig.add_subplot(1,3,3)
ax.plot(u_field_num[0], color='blue')
ax.plot(u_field_surr[0], color='red', alpha=0.5)
ax.set_xlabel('x')
ax.title.set_text('Final')

plt.savefig(os.getcwd() + '/Images/' + run_name + '.png')
run.save(os.getcwd() + '/Images/' + run_name + '.png', 'output')

end_time = time.time()
print(end_time - start_time)