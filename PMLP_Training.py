#Zi Xian Leong
#zxnleong@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import fnmatch
import math
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib
from numpy.matlib import repmat
from time import time


X_Train = np.load('data/for_DL/X_Train.npy')
y_Train = np.load('data/for_DL/y_Train.npy')
X_Val = np.load('data/for_DL/X_Val.npy')
y_Val = np.load('data/for_DL/y_Val.npy')
X_test  = np.load('data/for_DL/X_test.npy')
y_test = np.load('data/for_DL/y_test.npy')

print("X_Train's shape",X_Train.shape)
print("y_Train's shape",y_Train.shape)
print("X_Val's shape",X_Val.shape)
print("y_Val's shape",y_Val.shape)
print("X_test's shape",X_test.shape)
print("y_test's shape",y_test.shape)

def EarthDistanceLoss(prediction,ground_truth):
    '''
    distance formula
    input is [N,3]
    output is [N,3]
    '''
    if len(prediction.shape) == 1:
            prediction = prediction[None,:]
            ground_truth = ground_truth[None,:]
    
    predx = prediction[:,0] #(BS)
    predy = prediction[:,1] #(BS)
    predz = prediction[:,2] #(BS)
    
    gtx = ground_truth[:,0] #(BS)
    gty = ground_truth[:,1] #(BS)
    gtz = ground_truth[:,2] #(BS)

    distance_BS = torch.mean(torch.sqrt((predx-gtx)**2 + (predy-gty)**2 + (predz-gtz)**2))

    return distance_BS

def load_batch(xtrain,ytrain, batch_size=1,random=True):
    '''
    batch_size must be factor of len(dataset)
    '''
    
    # xtrain2 = xtrain[1]
    if len(xtrain) == len(ytrain):
        total_samples = len(xtrain)
        n_batches = int(total_samples/ batch_size)
        
        if random==True:
            indices = np.random.choice(total_samples,size=total_samples,replace=False)
        else:
            indices = np.arange(0,total_samples)
        
        for i in range(0, total_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_x1 = xtrain[batch_indices]
            batch_y = ytrain[batch_indices]
            
            yield batch_x1,batch_y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(device); print(torch.cuda.get_device_name())

def to_np(arr):
    return arr.detach().cpu().numpy()

#Convert all to torch tensor
X_train_torch = torch.tensor(X_Train).to(device)
X_valid_torch = torch.tensor(X_Val).to(device)
y_train_torch = torch.tensor(y_Train).to(device)
y_valid_torch = torch.tensor(y_Val).to(device)
X_test_torch = torch.tensor(X_test).to(device)

print(X_train_torch.shape)
print(X_valid_torch.shape)
print(y_train_torch.shape)
print(y_valid_torch.shape)
print(X_test_torch.shape)

# probabilistic MLP 
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3_mean = nn.Linear(200, 2)
        self.fc3_mean_depth = nn.Linear(200,1)
        self.fc3_log_std = nn.Linear(200, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean_rest = self.fc3_mean(x)
        mean_depth = torch.relu(self.fc3_mean_depth(x))
        log_std = self.fc3_log_std(x)
        mean = torch.cat((mean_rest,mean_depth),axis=-1)

        return mean, log_std

# USE THIS Probabilistic MLP 
torch.manual_seed(155)
np.random.seed(155)
num_epochs = 500
BS = 32
total_samples = X_Train.shape[0]
print(total_samples)
#total number of batches
n_batches = int(total_samples/ BS)
#print out progress after every x batches (e.g., after 10 batches)
print_batches = int(n_batches / 10)

run = 'good'
PMLP_model = MyModel(8, 3).to(device).double()

#Optimizer
lr = 0.0005
optimizer = torch.optim.Adam(PMLP_model.parameters(), lr = lr, betas=(0.9,0.9))
mseloss = nn.MSELoss()

#Main loop
main_log=dict()
record_Loss=[]
val_record_Loss = []
val_spec_record_Loss = []
start_time_main_loop = time()
for epoch in range(num_epochs):
    print('\n')
    print('''RUN {}, Epoch {} '''.format(run,epoch))
    PMLP_model.train()
    batch_Loss = []
    #start time for epoch
    start_time_epoch = time()
    for batch_i, (X_batch,y_batch) in enumerate(load_batch((X_train_torch),y_train_torch,batch_size=BS, random=True)):
        #zero the parameter gradients
        optimizer.zero_grad()
        #predict location
        # prediction = PMLP_model(X_batch)

        # Forward pass
        mean, log_std = PMLP_model(X_batch)
        std = torch.exp(log_std)

        # Negative log-likelihood loss for Gaussian distribution
        Loss = torch.mean(torch.log(std) + 0.5 * ((y_batch - mean) / std)**2)

        #Backpropagate
        Loss.backward()
        #Apply optimizer
        optimizer.step()

        ### Get batch distance loss
        samples_batchpred = torch.randn((1000, *mean.shape), device=mean.device) * std + mean
        mean_samples_batch = torch.mean(samples_batchpred, dim=0)
        batch_distance_Loss = EarthDistanceLoss(mean_samples_batch,y_batch)

        #Print out batch progress
        if batch_i % print_batches == 0:
            # print('[%d/%d], Batch:[%d/%d]\t Gaussian NLL Loss: %.9f ' % (epoch,num_epochs,batch_i,n_batches,Loss.item()))
            print('[%d/%d], Batch:[%d/%d]\t Earth Distance Loss: %.9f ' % (epoch,num_epochs,batch_i,n_batches,batch_distance_Loss.item()))
        #Save batch losses
        batch_Loss.append(batch_distance_Loss.item())
    #end time for epoch
    end_time_epoch = time()
    diff_time_epoch = end_time_epoch - start_time_epoch

    #Compute epoch losses
    epoch_Loss = np.mean(batch_Loss)
    
    #save epoch progress
    record_Loss.append(epoch_Loss)

    #save into dictionary
    main_log['loss'] = record_Loss
    
    #Print out Epoch Progress
    print('EPOCH %d Training -> Earth Distance Loss: %.9f ' %(epoch,epoch_Loss))
    print('EPOCH time taken: {} seconds'.format(diff_time_epoch))
    
    #########################################
    ############### Validation ##############
    #########################################
    with torch.no_grad():
        PMLP_model.eval()
        # val_prediction = PMLP_model(X_valid_torch)
        # val_Loss = EarthDistanceLoss(val_prediction,y_valid_torch)
        # print('EPOCH %d Validation -> EarthLoss: %.9f' %(epoch, val_Loss.item()))
        mean_valpred, log_std_valpred = PMLP_model(X_valid_torch)
        std_valpred = torch.exp(log_std_valpred)
        # Sample from Gaussian distribution
        samples_valpred = torch.randn((1000, *mean_valpred.shape), device=mean_valpred.device) * std_valpred + mean_valpred
        mean_samples = torch.mean(samples_valpred, dim=0)
    val_Loss = EarthDistanceLoss(mean_samples,y_valid_torch)
    print('EPOCH %d Validation -> EarthLoss: %.9f' %(epoch, val_Loss.item()))

    # #Save only the lowest validation loss score
    if epoch > 0:
        if val_Loss.item() < np.min(val_record_Loss):
            print('Saving lowest validation loss of: ',val_Loss.item())
            #save weights
            torch.save(PMLP_model.state_dict(),'weights_logs/GoodRun_Combined_LV.pth')


    #save into dictionary
    val_record_Loss.append(val_Loss.item())
    main_log['val_Loss'] = val_record_Loss

#end time main loop
end_time_main_loop = time()
diff_time_all = end_time_main_loop  - start_time_main_loop 
print('RUN good, Total time taken: {} seconds'.format(diff_time_all))
main_log['runtime'] = diff_time_all

#save log
np.save('weights_logs/GoodRun_Combined_log.npy',main_log)










