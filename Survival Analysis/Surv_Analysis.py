#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import glob
from sklearn.model_selection import train_test_split
from dataset import make_ds, make_ds_test
from monai.data import DataLoader
import matplotlib.pyplot as plt
import torch
from train_reconstruction import swinunetplussig, lossf
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from pycox.models import LogisticHazard
import torchtuples as tt
from pycox.evaluation import EvalSurv
from sklearn.metrics import roc_curve, roc_auc_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

df_clinic = pd.read_csv("/home/maryam/Surr_prognosis/surv_brats20.csv")
feature_tum = pd.read_csv("/home/maryam/Surr_prognosis/features_tum.csv")
feature_surr1 = pd.read_csv("/home/maryam/Surr_prognosis/features_surr1.csv")
feature_surr2 = pd.read_csv("/home/maryam/Surr_prognosis/features_surr2.csv")

df_total_train_val, df_total_test = train_test_split(
   df_clinic,
   test_size=0.2,
   stratify=df_clinic['event_1year'],
   random_state=121274  # for reproducibility
)
test_indices = df_total_test.index


validation_size = 0.10 / 0.80 
df_total_train, df_total_val = train_test_split(
   df_total_train_val,
   test_size=validation_size,
   stratify=df_total_train_val['event_1year'],
   random_state=121274
)
train_indices = df_total_train.index
val_indices = df_total_val.index

class survnet1(torch.nn.Module):
  
   def __init__(self, in_features, out_features):
             
       super(survnet1, self).__init__()


       self.fc1 = torch.nn.Linear(in_features, 10000)
       self.fc2 = torch.nn.Linear(10000, 1000)
       self.fc3 = torch.nn.Linear(1000, 100)
       self.fc4 = torch.nn.Linear(100, out_features)
              
       self.relu = torch.nn.ReLU()
       self.BN1 = torch.nn.BatchNorm1d(10000)
       self.BN2 = torch.nn.BatchNorm1d(1000)
       self.BN3 = torch.nn.BatchNorm1d(100)


       self.dropout = torch.nn.Dropout(0.3)
      
  
   def forward(self, imfeat_train):
      
       x1 = self.fc1(imfeat_train)
      
       x2 = self.relu(x1)
      
       x3 = self.BN1(x2)
      
       x4 = self.dropout(x3)
              
       x5 = self.fc2(x4)
      
       x6 = self.relu(x5)
      
       x7 = self.BN2(x6)
      
       x8 = self.dropout(x7)
      
       x9 = self.fc3(x8)

       x10 = self.relu(x9)
      
       x11 = self.BN3(x10)
      
       x12 = self.dropout(x11)

       x13 = self.fc4(x12)
              
       return x13

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##############
# Tumor Alone#
##############
np.random.seed(121274)
_ = torch.manual_seed(121274)

features_train = feature_tum.loc[train_indices]
features_val = feature_tum.loc[val_indices]
features_test = feature_tum.loc[test_indices]

scaler = StandardScaler()

imfeat_train = scaler.fit_transform(features_train).astype('float32')
imfeat_val = scaler.transform(features_val).astype('float32')
imfeat_test = scaler.transform(features_test).astype('float32')

get_target = lambda label_d: (label_d['time_1year'].values.astype(int), label_d['event_1year'].values.astype(int))


num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)


target_train = labtrans.fit_transform(*get_target(df_total_train))
target_val = labtrans.transform(*get_target(df_total_val))
target_test = labtrans.transform(*get_target(df_total_test))


train = tt.tuplefy(imfeat_train, target_train)
val = tt.tuplefy(imfeat_val, target_val)
test = tt.tuplefy(imfeat_test, target_test)

durations_train, events_train = get_target(df_total_train)
durations_val, events_val = get_target(df_total_val)
durations_test, events_test = get_target(df_total_test)

import gc
tensor = None
gc.collect()
torch.cuda.empty_cache()

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet1(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights("/home/maryam/Surr_prognosis/codes/tum/tumor_net1_0.56.pt")
# batch_size = 3
# epochs = 50
# callbacks = [tt.cb.EarlyStopping(patience=10)]
# log = model.fit(*train, batch_size, epochs, callbacks, True, val_data = val)

x_train = imfeat_train
x_val = imfeat_val
x_test = imfeat_test

surv_train_disc = model.predict_surv_df(x_train)
surv_val_disc = model.predict_surv_df(x_val)
surv_test_disc = model.predict_surv_df(x_test)


surv_train_cont = model.interpolate(40).predict_surv_df(x_train)
surv_val_cont = model.interpolate(40).predict_surv_df(x_val)
surv_test_cont = model.interpolate(40).predict_surv_df(x_test)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd_test = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd_test}')


ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


ev_val = EvalSurv(surv_val_cont, durations_val, events_val, censor_surv='km')
ctd_val = ev_val.concordance_td('antolini')
print(f'C_td Score for validation: {ctd_val}')

def bootstrap_cindex(surv, durations, events, n_iterations=1000):
    cindexs = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(durations), len(durations)) 
        ctd = EvalSurv(surv[indices], durations[indices], events[indices], censor_surv='km').concordance_td('antolini')
        cindexs.append(ctd)
    sorted_cs = np.array(cindexs)
    sorted_cs.sort()
    conf_int = np.percentile(sorted_cs , [2.5, 97.5])
    return conf_int

print(f"95 CI Cindex totaltest:{bootstrap_cindex(surv_test_cont, durations_test, events_test)}")

risk_scores2 = model.predict(x_test)
average_risk_scores2 = np.mean(risk_scores2, axis=1)

y_true_binary2 = events_test

fpr_test, tpr_test, _ = roc_curve(y_true_binary2, average_risk_scores2)
auc_test = roc_auc_score(y_true_binary2, average_risk_scores2)

risk_scores3 = model.predict(x_train)
average_risk_scores3 = np.mean(risk_scores3, axis=1)

y_true_binary3 = events_train 

fpr_train, tpr_train, _ = roc_curve(y_true_binary3, average_risk_scores3)
auc_train = roc_auc_score(y_true_binary3, average_risk_scores3)

plt.figure()
plt.plot(fpr_test, tpr_test, label=f" AUC test = {auc_test:.2f}")
plt.plot(fpr_train, tpr_train, label=f" AUC Train = {auc_train:.2f}")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

def bootstrap_auc(y_true, y_score, n_iterations=1000):
    aucs = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true)) 
        sample_auc = roc_auc_score(y_true[indices], y_score[indices])
        aucs.append(sample_auc)
    sorted_auc = np.array(aucs)
    sorted_auc.sort()
    conf_int = np.percentile(sorted_auc , [2.5, 97.5])
    return conf_int

print(f"95 CI AUC test:{bootstrap_auc(y_true_binary2, average_risk_scores2)}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####################
#Tumor +5mm Dilation#
#####################
np.random.seed(121274)
_ = torch.manual_seed(121274)

features_train = feature_surr1.loc[train_indices]
features_val = feature_surr1.loc[val_indices]
features_test = feature_surr1.loc[test_indices]

scaler = StandardScaler()

imfeat_train = scaler.fit_transform(features_train).astype('float32')
imfeat_val = scaler.transform(features_val).astype('float32')
imfeat_test = scaler.transform(features_test).astype('float32')

get_target = lambda label_d: (label_d['time_1year'].values.astype(int), label_d['event_1year'].values.astype(int))


num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)


target_train = labtrans.fit_transform(*get_target(df_total_train))
target_val = labtrans.transform(*get_target(df_total_val))
target_test = labtrans.transform(*get_target(df_total_test))


train = tt.tuplefy(imfeat_train, target_train)
val = tt.tuplefy(imfeat_val, target_val)
test = tt.tuplefy(imfeat_test, target_test)

durations_train, events_train = get_target(df_total_train)
durations_val, events_val = get_target(df_total_val)
durations_test, events_test = get_target(df_total_test)

import gc
tensor = None
gc.collect()
torch.cuda.empty_cache()

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet1(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights("/home/maryam/Surr_prognosis/codes/sur1/sur1_net1_0.63.pt")

# batch_size = 3
# epochs = 50
# callbacks = [tt.cb.EarlyStopping(patience=10)]
# log = model.fit(*train, batch_size, epochs, callbacks, True, val_data = val)

x_train = imfeat_train
x_val = imfeat_val
x_test = imfeat_test

surv_train_disc = model.predict_surv_df(x_train)
surv_val_disc = model.predict_surv_df(x_val)
surv_test_disc = model.predict_surv_df(x_test)


surv_train_cont = model.interpolate(40).predict_surv_df(x_train)
surv_val_cont = model.interpolate(40).predict_surv_df(x_val)
surv_test_cont = model.interpolate(40).predict_surv_df(x_test)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd_test = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd_test}')


ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


ev_val = EvalSurv(surv_val_cont, durations_val, events_val, censor_surv='km')
ctd_val = ev_val.concordance_td('antolini')
print(f'C_td Score for validation: {ctd_val}')

def bootstrap_cindex(surv, durations, events, n_iterations=1000):
    cindexs = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(durations), len(durations)) 
        ctd = EvalSurv(surv[indices], durations[indices], events[indices], censor_surv='km').concordance_td('antolini')
        cindexs.append(ctd)
    sorted_cs = np.array(cindexs)
    sorted_cs.sort()
    conf_int = np.percentile(sorted_cs , [2.5, 97.5])
    return conf_int

print(f"95 CI Cindex totaltest:{bootstrap_cindex(surv_test_cont, durations_test, events_test)}")

risk_scores2 = model.predict(x_test)
average_risk_scores2 = np.mean(risk_scores2, axis=1)

y_true_binary2 = events_test

fpr_test, tpr_test, _ = roc_curve(y_true_binary2, average_risk_scores2)
auc_test = roc_auc_score(y_true_binary2, average_risk_scores2)

risk_scores3 = model.predict(x_train)
average_risk_scores3 = np.mean(risk_scores3, axis=1)

y_true_binary3 = events_train 

fpr_train, tpr_train, _ = roc_curve(y_true_binary3, average_risk_scores3)
auc_train = roc_auc_score(y_true_binary3, average_risk_scores3)

plt.figure()
plt.plot(fpr_test, tpr_test, label=f" AUC test = {auc_test:.2f}")
plt.plot(fpr_train, tpr_train, label=f" AUC Train = {auc_train:.2f}")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

def bootstrap_auc(y_true, y_score, n_iterations=1000):
    aucs = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true)) 
        sample_auc = roc_auc_score(y_true[indices], y_score[indices])
        aucs.append(sample_auc)
    sorted_auc = np.array(aucs)
    sorted_auc.sort()
    conf_int = np.percentile(sorted_auc , [2.5, 97.5])
    return conf_int

print(f"95 CI AUC test:{bootstrap_auc(y_true_binary2, average_risk_scores2)}")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####################
#Tumor +1cm dilation#
#####################
np.random.seed(121274)
_ = torch.manual_seed(121274)

features_train = feature_surr2.loc[train_indices]
features_val = feature_surr2.loc[val_indices]
features_test = feature_surr2.loc[test_indices]

scaler = StandardScaler()

imfeat_train = scaler.fit_transform(features_train).astype('float32')
imfeat_val = scaler.transform(features_val).astype('float32')
imfeat_test = scaler.transform(features_test).astype('float32')

get_target = lambda label_d: (label_d['time_1year'].values.astype(int), label_d['event_1year'].values.astype(int))


num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)


target_train = labtrans.fit_transform(*get_target(df_total_train))
target_val = labtrans.transform(*get_target(df_total_val))
target_test = labtrans.transform(*get_target(df_total_test))


train = tt.tuplefy(imfeat_train, target_train)
val = tt.tuplefy(imfeat_val, target_val)
test = tt.tuplefy(imfeat_test, target_test)

durations_train, events_train = get_target(df_total_train)
durations_val, events_val = get_target(df_total_val)
durations_test, events_test = get_target(df_total_test)

import gc
tensor = None
gc.collect()
torch.cuda.empty_cache()

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet1(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights("/home/maryam/Surr_prognosis/codes/sur2/sur2_net1_0.58.pt")

# batch_size = 3
# epochs = 50
# callbacks = [tt.cb.EarlyStopping(patience=10)]
# log = model.fit(*train, batch_size, epochs, callbacks, True, val_data = val)

x_train = imfeat_train
x_val = imfeat_val
x_test = imfeat_test

surv_train_disc = model.predict_surv_df(x_train)
surv_val_disc = model.predict_surv_df(x_val)
surv_test_disc = model.predict_surv_df(x_test)

surv_train_cont = model.interpolate(40).predict_surv_df(x_train)
surv_val_cont = model.interpolate(40).predict_surv_df(x_val)
surv_test_cont = model.interpolate(40).predict_surv_df(x_test)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd_test = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd_test}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


ev_val = EvalSurv(surv_val_cont, durations_val, events_val, censor_surv='km')
ctd_val = ev_val.concordance_td('antolini')
print(f'C_td Score for validation: {ctd_val}')

def bootstrap_cindex(surv, durations, events, n_iterations=1000):
    cindexs = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(durations), len(durations)) 
        ctd = EvalSurv(surv[indices], durations[indices], events[indices], censor_surv='km').concordance_td('antolini')
        cindexs.append(ctd)
    sorted_cs = np.array(cindexs)
    sorted_cs.sort()
    conf_int = np.percentile(sorted_cs , [2.5, 97.5])
    return conf_int

print(f"95 CI Cindex totaltest:{bootstrap_cindex(surv_test_cont, durations_test, events_test)}")

risk_scores2 = model.predict(x_test)
average_risk_scores2 = np.mean(risk_scores2, axis=1)

y_true_binary2 = events_test

fpr_test, tpr_test, _ = roc_curve(y_true_binary2, average_risk_scores2)
auc_test = roc_auc_score(y_true_binary2, average_risk_scores2)

risk_scores3 = model.predict(x_train)
average_risk_scores3 = np.mean(risk_scores3, axis=1)

y_true_binary3 = events_train 

fpr_train, tpr_train, _ = roc_curve(y_true_binary3, average_risk_scores3)
auc_train = roc_auc_score(y_true_binary3, average_risk_scores3)

plt.figure()
plt.plot(fpr_test, tpr_test, label=f" AUC test = {auc_test:.2f}")
plt.plot(fpr_train, tpr_train, label=f" AUC Train = {auc_train:.2f}")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

def bootstrap_auc(y_true, y_score, n_iterations=1000):
    aucs = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true)) 
        sample_auc = roc_auc_score(y_true[indices], y_score[indices])
        aucs.append(sample_auc)
    sorted_auc = np.array(aucs)
    sorted_auc.sort()
    conf_int = np.percentile(sorted_auc , [2.5, 97.5])
    return conf_int

print(f"95 CI AUC test:{bootstrap_auc(y_true_binary2, average_risk_scores2)}")

