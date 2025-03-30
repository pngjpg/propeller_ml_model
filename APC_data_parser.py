# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:50:55 2024

@author: Brian
"""

import os
import re
import pandas as pd
import seaborn as sns


#%%

cwd = os.getcwd()+'\\APC CFD Data\\PERFILES2 - Culled'

structured_data = list()


for filename in os.listdir(cwd):
    with open(cwd+"\\"+filename) as f:
        print(filename)
        
        diam = filename.split("_")[1].split("x")[0].replace("-", ".")
        pitch = filename.split("_")[1].split("x")[1].split("E")[0].replace("-", ".")
        
        data = f.read().split("\n")[1:-1]
        
        in_header = True
        skip_lines= 20
        current_rpm = 1000
        for line in data:
            if skip_lines:
                skip_lines -= 1
            else:
                if "PROP RPM" in line:
                    current_rpm = line.split("=")[-1]
                    skip_lines = 3
                else:
                    if not "." in line:
                        pass
                    else:
                        parsed_line = list(filter(lambda a: a != "", line.split(" ")))
                        # print(parsed_line)
                        if not len(parsed_line) == 15:
                            pass
                        else:
                            output_data = [
                                float(parsed_line[1]), # Advance Ratio
                                float(parsed_line[3]),
                                float(parsed_line[4]),
                                float(parsed_line[2]),
                                ]
                            structured_data.append([float(diam), float(pitch), int(current_rpm)/1000] + output_data)
    # break
df = pd.DataFrame(structured_data, columns=["diam", "pitch", "rpm", "J", "CT", "CP", "eta"])
df.to_csv("output_APC.csv", index=False)

#%% Plotting

pairplot = sns.pairplot(df, plot_kws={'s': 3, 'color': 'black', 'alpha': 0.2})

#%% Set Test Parameters

diam_sel = 12
pitch_sel = 10

# RPMs: {3015, 3041, 3987, 4032, 4969, 5019, 5962, 5995}

rpm_sel = 5.0

#%% PyTorch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
# import seaborn as sns
import pandas as pd

seed = 42
torch.manual_seed(seed)

class MLPcondensed(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self):
        
        nInput=4
        nHidden1=128
        nHidden2=128
        nOutput=3
        
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nInput, nHidden1),
            nn.SiLU(),
            nn.Linear(nHidden1, nHidden1),
            nn.SiLU(),
            nn.Linear(nHidden1, nHidden2),
            nn.SiLU(),
            nn.Linear(nHidden2, nOutput),
            nn.SiLU()
        )

    def forward(self, x):
        return(self.layers(x))

model = MLPcondensed()

class nonLinearRegressionData(Dataset):
    '''
    Custom 'Dataset' object for our regression data.
    Must implement these functions: __init__, __len__, and __getitem__.
    '''

    def __init__(self):
        
        # plotted = df[df["diam"] == diam_sel]
        # plotted = plotted[plotted["pitch"] == pitch_sel]
        # plotted = plotted[plotted["rpm"] == rpm_sel]
        
        self.xObs = torch.tensor(df[["diam", "pitch", "rpm", "J"]].astype(np.float32).values)
        self.yObs = torch.tensor(df[["CT", "CP", "eta"]].astype(np.float32).values)
        
        # self.xObs = torch.tensor(df[["diam", "pitch", "rpm", "J"]].astype(np.float32).values)
        # self.yObs = torch.tensor(df[["eta"]].astype(np.float32).values)

    def __len__(self):
        return(len(self.xObs))

    def __getitem__(self, idx):
        return(self.xObs[idx], self.yObs[idx])

# instantiate Dataset object for current training data
d = nonLinearRegressionData()

# instantiate DataLoader
train_dataloader = DataLoader(d, batch_size=32 , shuffle=True)

#%% Training

# # loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MSELoss()
# # loss = loss_fn(output, label)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# # optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2, max_iter=40)

# for n in range(20):
    
#     current_loss = 0.0
    
#     for i, data in enumerate(train_dataloader, 0):
        
#         # Get inputs
#         inputs, targets = data
#         # targets = targets.reshape((targets.shape[0], 1))
#         # Zero the gradients
#         optimizer.zero_grad()
#         # Perform forward pass (make sure to supply the input in the right way)
#         outputs = model(inputs)
#         # Compute loss
#         loss = loss_fn(outputs, targets)
#         # Perform backward pass
#         loss.backward()
#         # Perform optimization
#         optimizer.step()
#         # Print statistics
#         current_loss += loss.item()
        
    
#     if n % 1 == 0:
#     # if n % 4000 == 0:
#     #     for g in optimizer.param_groups:
#     #         g['lr'] = g['lr']/10
    
#     # if n == 9000:
#     #     for g in optimizer.param_groups:
#     #         g['lr'] = g['lr']/100

    
#         model.eval()
#         # print(model(torch.Tensor(np.array([diam_sel, pitch_sel, rpm_sel, 0.8]).astype(np.float32))))
        
#         from sklearn.metrics import mean_squared_error, r2_score
        
#         test_data = df[df["diam"] == diam_sel]
#         test_data = test_data[test_data["pitch"] == pitch_sel]
#         test_data = test_data[test_data["rpm"] == rpm_sel]
        
#         with torch.no_grad():
#             outputs = model(torch.tensor(test_data[["diam","pitch", "rpm", "J"]].astype(np.float32).values))
#             predicted_labels = outputs.squeeze().tolist()
        
#         predicted_labels = np.array(predicted_labels)
#         test_targets = np.array(test_data[["CT", "CP", "eta"]])
        
#         mse = mean_squared_error(test_targets, predicted_labels)
#         r2 = r2_score(test_targets, predicted_labels)
#         # print("Mean Squared Error:", mse)
#         # print("R2 Score:", r2)
        
#         print(str(n) + ": " + str(round(current_loss,5)) + " MSE: " + str(round(mse,3)) + " R^2: " + str(round(r2,3)))

        
#         model.train()

#%% Load Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
# import seaborn as sns
import pandas as pd

model = MLPcondensed() # initialize your model class
model.load_state_dict(torch.load(os.getcwd()+"\\model_APC_1k_epoch.pth"))
    
#%% Plotting

def get_r_squared(y_true, y_fit):
    residuals = y_true - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

use_ml_model = True

diam_sel = 8
pitch_sel = 8

plotted = df[df["diam"] == diam_sel]
plotted = plotted[plotted["pitch"] == pitch_sel]
rpm_selection = set(plotted["rpm"].to_numpy())

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

def plot_rpm(rpm_sel, color):
    plotted_temp = plotted[plotted["rpm"] == rpm_sel]
    ax[0].scatter(plotted_temp["J"].tolist(), plotted_temp["CT"].tolist(), color = color)
    fit_model_x = plotted_temp["J"].tolist()
    fit_model_y = [model(torch.Tensor(np.array([diam_sel, pitch_sel, rpm_sel, model_J]).astype(np.float32)))[0].detach().numpy() for model_J in fit_model_x]
    ax[0].plot(fit_model_x, fit_model_y, color = color, label = f"RPM: {rpm_sel}, R^2: {round(get_r_squared(plotted_temp['CT'].to_numpy(), np.array(fit_model_y)),4)}")
    ax[0].set_title("CT")
    ax[0].set_xlabel("J")
    ax[0].legend(loc = "lower left")
    
    ax[1].scatter(plotted_temp["J"].tolist(), plotted_temp["CP"].tolist(), color = color)
    fit_model_y = [model(torch.Tensor(np.array([diam_sel, pitch_sel, rpm_sel, model_J]).astype(np.float32)))[1].detach().numpy() for model_J in fit_model_x]
    ax[1].plot(fit_model_x, fit_model_y, color = color, label = f"RPM: {rpm_sel}, R^2: {round(get_r_squared(plotted_temp['CP'].to_numpy(), np.array(fit_model_y)),4)}")
    ax[1].set_title("CP")
    ax[1].set_xlabel("J")
    ax[1].legend(loc = "lower left")
    
    ax[2].scatter(plotted_temp["J"].tolist(), plotted_temp["eta"].tolist(), color = color)
    fit_model_y = [model(torch.Tensor(np.array([diam_sel, pitch_sel, rpm_sel, model_J]).astype(np.float32)))[2].detach().numpy() for model_J in fit_model_x]
    ax[2].plot(fit_model_x, fit_model_y, color = color, label = f"RPM: {rpm_sel}, R^2: {round(get_r_squared(plotted_temp['eta'].to_numpy(), np.array(fit_model_y)),4)}")
    ax[2].set_title("Eta")
    ax[2].set_xlabel("J")
    ax[2].legend(loc = "lower center")

cmap = mpl.colormaps["jet"]

for rpm in range(len(rpm_selection)):
    plot_rpm(rpm+1, cmap( (rpm)/(len(rpm_selection)) ))

fig.suptitle(f"Diam: {diam_sel}, Pitch: {pitch_sel}, RPM: {rpm_selection}")

