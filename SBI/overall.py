import matplotlib.pyplot as plt
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbi import analysis, utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from classification import make_preds_accumul

seed = 0
torch.manual_seed(seed);

from pyriemann.estimation import Xdawn
import sys

sys.path.append('D:/s.velut/Documents/These/Protheus_PHD/Scripts')
from Alignments.riemannian import compute_riemannian_alignment
from EEG2CodeKeras import EEG2Code
import keras
from EEGModels import EEGNet
from pyriemann.utils.mean import mean_covariance
from Alignments.covariance import compute_covariances
from Alignments.riemannian import compute_riemannian_alignment
from imblearn.under_sampling import RandomUnderSampler
from _utils import make_preds_accumul_aggresive
from sklearn.metrics import balanced_accuracy_score
from sbi.neural_nets.embedding_nets import CNNEmbedding
from zuko.nn import MLP
from torch.nn import RNN, LSTM
import keras


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyntbci.utilities import correlation
from collections import OrderedDict
from typing import Tuple
import pyntbci
import scipy.io
from simulated_cvep.forward_model import gaussian, fwd

from simulated_cvep import forward_model

sys.path.insert(0,"D:/s.velut/Documents/These/Protheus_PHD/Scripts/Sim_Trans_Link/")
from STLDataLoader import STLDataLoader
from utils import balance

class SummaryCNN(nn.Module):
    """
    Represents a CNN with growing linear module as end layer.
    """

    def __init__(self,
        in_features: int | list[int] | tuple[int],
        out_features: int,
        n_channel_input: int,
        ) -> None:
        """
        Initialize the growing MLP.

        Parameters
        ----------
        in_features : int | list | tuple
            Number of input features.
        out_features : int
            Number of output features.
        n_channel_input : int
            Number of channel .
        number_hidden_layers : int
            Number of hidden layers.
        activation : nn.Module
            Activation function.
        use_bias : bool
            Whether to use bias in layers.
        flatten : bool
            Whether to flatten the input before passing it through the network.
        device : Optional[torch.device]
            Device to use for computation.
        """

        super().__init__(
        )

        
        self.n_channel_input = n_channel_input

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1,16, kernel_size=(self.n_channel_input,1), stride=(1,1)))
        self.layers.append(nn.BatchNorm2d(16))
        self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1)))
        self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Conv2d(16,8, kernel_size=(1,32), stride=(1,1), dilation=(1,2), padding=(0,31)))
        self.layers.append(nn.BatchNorm2d(8))
        self.layers.append(nn.LeakyReLU(0.3))
        self.layers.append(nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=(0,0)))
        self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Conv2d(8,4, kernel_size=(5,5), stride=(1,1), dilation=(2,2), padding=(4,4)))
        self.layers.append(nn.BatchNorm2d(4))
        self.layers.append(nn.LeakyReLU(0.3))
        self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,0)))
        self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(444, 256))
        self.layers.append(nn.Linear(256, out_features))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, n_channels, window_size)
        for layer in self.layers:
            # print(layer,x.shape)
            x = layer(x)

        return x

def get_train_test_data(Xt, Yt, domainst, labels_code, clf_name, n_class=5, n_cal=4, window_size=0.35, freqwise=500):
    # Initialisation
    nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

    # Get the training data 
    X_train = Xt[:nb_sample_cal]
    Y_train = Yt[:nb_sample_cal]
    domains_train = domainst[:nb_sample_cal]
    X_test = Xt[nb_sample_cal:]
    Y_test = Yt[nb_sample_cal:]
    domains_test = domainst[nb_sample_cal:]
    labels_code_test = labels_code[(n_class*n_cal):]
    # X_std = Xt[:nb_sample_cal].std(axis=0)
    # X_train = X_train/(X_std + 1e-8)
    # X_test = X_test/(X_std + 1e-8)

    X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)

    return X_train, Y_train, X_test, Y_test, domains_train, domains_test, labels_code_test

def full_preprocessed_data(path,participant):
    return np.load(path+"full_preprocess_data_"+participant+".npy")

def simulator_model(theta,code = None,codes=None, only_template=False):
    pr = 60
    length_vep=0.35
    # V = pyntbci.stimulus.make_gold_codes()[subset][0]
    # V = pyntbci.stimulus.modulate(V)
    V = codes.copy()
    
    # V = np.zeros((nclass,ntimes))
    # for k in range(nclass):
    #     code_pos = 0
    #     count = 0
    #     for idx in range(ntimes):
    #         if idx/fs >= (code_pos+1)/60:
    #             code_pos += 1 
    #         V[k,count] = int(codes[k][code_pos])
    #         count += 1
    # V = torch.Tensor(V)
    V = torch.repeat_interleave(torch.tensor(V), int(500 / pr),dim=1)
    # V = torch.repeat_interleave(torch.tensor(np.tile(V,5)), int(500 / pr),dim=1)
    # V = torch.Tensor(np.concatenate([V[:,0:1],np.diff(V)],axis=1)%255)
    # print(V[0,:40])
    t = torch.arange(0, length_vep, 1 / 500)

    sim_vep, sim_signal = fwd(t, V, theta, 0.5)

    noise=0
    snr = 0.1
    sig_avg = torch.mean(sim_signal**2,axis=1)
    sig_avg_db = 10 * np.log10(sig_avg)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - 1/snr
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(torch.mean(noise_avg)), sim_signal.shape)

    if code is None:
        code = np.arange(0,V.shape[0],1)
    if only_template==False:
        return sim_signal[code]+noise[code]
    else:
        return sim_signal[code]

class SummaryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SummaryRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        rnn_out, hidden = self.rnn(torch.unsqueeze(input,dim=1))
        output = self.h2o(rnn_out)
        output = self.relu(output)

        return output.squeeze(1)
    
def cross_validate(X_emp, X_syn, y, V,y_syn=None, n_folds=4):
    if y_syn is None:
        y_syn = y.copy()
    accuracy = np.zeros((4, n_folds))
    folds = np.arange(n_folds).repeat(X_emp.shape[0] / n_folds)[:X_emp.shape[0]]
    folds_sim = np.arange(n_folds).repeat(X_syn.shape[0] / n_folds)[:X_syn.shape[0]]
    for i_fold in range(n_folds):
        
        X_emp_trn = X_emp[folds != i_fold, :, :]
        X_emp_tst = X_emp[folds == i_fold, :, :]

        X_syn_trn = X_syn[folds_sim != i_fold, :, :]
        X_syn_tst = X_syn[folds_sim == i_fold, :, :]

        y_trn = y[folds != i_fold]
        y_trn_syn = y_syn[folds_sim != i_fold]
        y_tst = y[folds == i_fold]
        y_tst_syn = y_syn[folds_sim == i_fold]
        
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=500, event="id", onset_event=False, encoding_length=0.35)

        # Emp to emp
        rcca.fit(X_emp_trn, y_trn)
        yh_tst = rcca.predict(X_emp_tst)
        accuracy[0, i_fold] = np.mean(yh_tst == y_tst)

        # Emp to syn
        rcca.fit(X_emp_trn, y_trn)
        yh_tst = rcca.predict(X_syn_tst)
        accuracy[1, i_fold] = np.mean(yh_tst == y_tst_syn)

        # Syn to emp
        rcca.fit(X_syn_trn, y_trn_syn)
        yh_tst = rcca.predict(X_emp_tst)
        accuracy[2, i_fold] = np.mean(yh_tst == y_tst)

        # Syn to syn
        rcca.fit(X_syn_trn, y_trn_syn)
        yh_tst = rcca.predict(X_syn_tst)
        accuracy[3, i_fold] = np.mean(yh_tst == y_tst_syn)
    return accuracy


def get_alldata(subj):
    to_remove = []
    dl = STLDataLoader('D:/s.velut/Documents/These/Protheus_PHD/Data/Dry_Ricker/', 1, 45, 0.35, 500, 60, "time_sample", [subj], 5)
    raw_data = dl.load_data()
    X, Y, domains, codesdict, labels_code = dl.get_epochs(raw_data)

    def changeEventID(events, event_id):
            new_dic = {}
            for k in event_id.keys():
                new_dic[k.split('_')[1]] = event_id[k]
                event_id[k] = int(k.split('_')[1])
            for i in range(len(events)):
                events[i][2] = int(new_dic[str(events[i][2])])
            
            return events, event_id 

    for idx in range(len(raw_data[0].annotations.description)):
        if (('boundary' in raw_data[0].annotations.description[idx]) or
            ('BURST' in raw_data[0].annotations.description[idx])):
            to_remove.append(idx)

    to_remove = np.array(to_remove)
    if len(to_remove) > 0:
        raw_data[0].annotations.delete(to_remove)
    temp_event,temp_event_id = mne.events_from_annotations(raw_data[0], event_id='auto', verbose=False)
    temp_event,temp_event_id = changeEventID(temp_event,temp_event_id)
    epochs = mne.Epochs(raw_data[0], temp_event, event_id=temp_event_id, tmin=0, 
                    tmax=2.2, baseline=(None, None), preload=True, verbose=False).resample(480)
    y = epochs.events[..., -1]

    return epochs, y, X, Y, np.array([codesdict[i] for i in range(len(codesdict))],dtype=np.uint8)

########### Get the data ############
participants = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10',\
                    'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',\
                    'P21','P22','P23','P24']
# participants = ["P1","P2","P3"]
path = 'D:/s.velut/Documents/These/Protheus_PHD/Data/Dry_Ricker'
n_class = 5
n_cal = 4
sfreq = 480
show_accuracy = True


parameters_all_ERP = {"P1":[0.13,0.177,0.235,-0.1,1.7,-0.9,0.01,0.025,0.032],
                  "P2":[0.157,0.21,0.28,-0.3,1.2,-1.2,0.01,0.03,0.02],
                  "P3":[0.107,0.188,0.278,-0.05,1.37,-0.65,0.02,0.02,0.03],
                  "P4":[0.135,0.175,0.24,-1.3,2.1,-1.0,0.02,0.035,0.02],
                  "P5":[0.12,0.187,0.255,-0.2,1.5,-0.7,0.02,0.027,0.03],
                  "P6":[0.15,0.21,0.285,-0.4,1.6,-1.3,0.02,0.04,0.03],
                  "P7":[0.12,0.185,0.31,-0.05,1.5,-1.2,0.02,0.03,0.04],
                  "P8":[0.13,0.183,0.28,-0.4,1.85,-0.7,0.01,0.03,0.05],
                  "P9":[0.125,0.16,0.23,-0.8,1.2,-0.6,0.02,0.03,0.03],
                  "P10":[0.13,0.21,0.31,-0.25,0.9,-0.1,0.02,0.03,0.01],
                  "P11":[0.135,0.174,0.23,-0.5,1.55,-1,0.01,0.026,0.027],
                  "P12":[0.17,0.22,0.31,-0.7,1.45,-0.45,0.015,0.03,0.04],
                  "P13":[0.17,0.235,0.3,-0.8,1.8,-0.5,0.025,0.03,0.02],
                  "P14":[0.11,0.198,0.295,-0.25,1.65,-0.7,0.03,0.03,0.02],
                  "P15":[0.15,0.2, 0.32, -0.1, 0.95, -0.55, 0.01, 0.018, 0.03],
                  "P16":[0.07,0.19, 0.285, -0.45, 1.75, -0.85, 0.03, 0.028, 0.03],
                  "P17":[0.11,0.178,0.235,-0.05,1.7,-0.9,0.01,0.025,0.03],
                  "P18":[0.17,0.215, 0.33, -0.7, 1.5, -0.8, 0.015, 0.04, 0.03],
                  "P19":[0.03,0.095, 0.175, -0.005, 1.45, -0.9, 0.01, 0.032, 0.04],
                  "P20":[0.11,0.2, 0.305, -0.002, 1.4, -0.95, 0.055, 0.03, 0.02],
                  "P21":[0.14,0.182,0.31,-0.85,2.25,-0.4,0.02,0.02,0.03],
                  "P22":[0.12,0.18,0.295,-0.4,1.55,-0.55,0.02,0.03,0.03],
                  "P23":[0.125,0.17,0.24,-0.3,0.8,-0.6,0.015,0.015,0.015],
                  "P24":[0.12,0.165,0.24,-1,2.1,-0.5,0.02,0.03,0.04]
}

# # get the data
# dl = STLDataLoader(path, 1, 45, 0.35, sfreq, 60, "time_sample", participants, 5)
# raw_data = dl.load_data()
# Xall, Yall, domainsall, codesdict, labels_code = dl.get_epochs(raw_data)
# trials,trial_labels = dl.get_trials(raw_data)
# # trials_preproc = trials[:,:10].copy()
# codes = np.array([codesdict[i] for i in range(len(codesdict))],dtype=np.uint8)

# X_preproc = np.array([full_preprocessed_data('/'.join(['D:/s.velut/Documents/These/Protheus_PHD/Data/STL',"ws"+str(0.35),"time_sample","data",'']),parti) for parti in participants])
# for i in range(Xall.shape[0]):
#     xdawn = Xdawn(nfilter=4,classes=[1],estimator='lwf')
#     X_std = Xall[i][:int(n_cal*n_class*(2.2-0.35)*sfreq)].std(axis=0)
#     temp_X = Xall[i][:int(n_cal*n_class*(2.2-0.35)*sfreq)]/(X_std + 1e-8)
#     temp_trial = trials[i]/(trials[i][:n_cal*n_class].std(axis=0) + 1e-8)
#     # trials[i] = trials[i]/(trials[i][:n_cal*n_class].std(axis=0) + 1e-8)
#     xdawn = xdawn.fit(temp_X,Yall[i][:int(n_cal*n_class*(2.2-0.35)*sfreq)])
#     print('transform trial')
#     temp_trial = xdawn.transform(temp_trial)
#     trials[i] = np.hstack([temp_trial,np.tile(np.mean(temp_trial,axis=0),(temp_trial.shape[0],1,1))])
#     trials[i] = compute_riemannian_alignment(trials[i], mean=None, dtype='real')


for ind_part in range(len(participants)):
    epochs,trial_labels,Xall,Yall,codes = get_alldata(participants[ind_part])
    y_true = trial_labels
    trials = epochs.get_data()[:,:,:-1]#/np.linalg.norm(epochs.get_data()[:,:,:-1])

    xdawn = Xdawn(nfilter=4,classes=[1],estimator='lwf')
    # temp_trial = trials/(trials[:4*5].std(axis=0) + 1e-8)
    # xdawn = xdawn.fit(temp_trial[:4*5],trial_labels[:int(4*5)])
    X_std = Xall[0][:int(4*5*(2.2-0.35)*sfreq)].std(axis=0)
    temp_X = Xall[0][:int(4*5*(2.2-0.35)*sfreq)]/(X_std + 1e-8)
    temp_trial = trials/(trials[:4*5].std(axis=0) + 1e-8)
    # trials[0] = trials[0]/(trials[0][:4*5].std(axis=0) + 1e-8)
    xdawn = xdawn.fit(temp_X,Yall[0][:int(4*5*(2.2-0.35)*sfreq)])
    print('transform trial')
    temp_trial = xdawn.transform(temp_trial)
    trials = np.hstack([temp_trial,np.tile(np.mean(temp_trial,axis=0),(temp_trial.shape[0],1,1))])
    trials = compute_riemannian_alignment(trials, mean=None, dtype='real')

    print("Participant ", participants[ind_part])
    # optimizer = keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
    # model = EEG2Code(windows_size = X_preproc[ind_part].shape[-1],
    #                     n_channel_input = X_preproc[ind_part].shape[-2],
    #                     optimizer=optimizer,
    #                     num_epochs=25)

    # X_train, Y_train, X_test, Y_test, domains_train, domains_test, labels_code_test = get_train_test_data(X_preproc[ind_part],
    #                                                                                             Yall[ind_part],
    #                                                                                             domainsall[ind_part],
    #                                                                                             labels_code,
    #                                                                                             clf_name="EEG2Code",
    #                                                                                             n_class=n_class,
    #                                                                                             n_cal=n_cal,
    #                                                                                             window_size=0.35,
    #                                                                                             freqwise=sfreq)


    # model = model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    # Y_pred = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
    # if show_accuracy:
    #     print("EEG2Code Test Balanced Accuracy: ", balanced_accuracy_score(Y_test, Y_pred))
    #     np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/training_balanced_accuracy_participant_"+str(ind_part)+".npy",balanced_accuracy_score(Y_test, Y_pred))

    # X_preproc_frame = np.array([full_preprocessed_data('/'.join(['D:/s.velut/Documents/These/Protheus_PHD/Data/STL',"ws"+str(0.35),"frame","data",'']),parti) for parti in participants[ind_part:ind_part+1]])
    # n_detection_inf = int(n_class*n_cal*(2.2-0.35)*60)
    # n_detection_sup = int((n_class*n_cal+1)*(2.2-0.35)*60)
    # X_detection = X_preproc_frame[0][n_detection_inf:n_detection_sup]
    # y_pred_detection = model.predict(X_detection)
    # detected_code = np.array([1 if (y >= 0.5) else 0 for y in y_pred_detection])

    # if show_accuracy:
    #     dl_frame = STLDataLoader(path, 1, 45, 0.35, sfreq, 60, "frame", [participants[ind_part]], 5)
    #     raw_data_frame = dl_frame.load_data()
    #     Xframe, Yframe, domainsframe, _, _ = dl_frame.get_epochs(raw_data_frame)
    #     Y_detection = Yframe[0][n_detection_inf:n_detection_sup]
    #     y_pred_detection = model.predict(X_preproc_frame[0])
    #     Y_pred_detection = np.array([1 if (y >= 0.5) else 0 for y in y_pred_detection])
    #     np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/frame_balanced_accuracy_participant_"+str(ind_part)+".npy",balanced_accuracy_score(Yframe[0], Y_pred_detection))

    detected_code = codes[trial_labels[n_class*n_cal]-1]
    ####### SBI ################
    embedding_net = SummaryRNN(input_size=1056, hidden_size=128, output_size=20)
    # embedding_net = SummaryCNN(in_features=888, out_features=20, n_channel_input=1)
    prior = utils.BoxUniform(low=torch.FloatTensor([0.07,0.10,0.23,-1.3,0.4,-1.3,0.01,0.01,0.01]), high=torch.FloatTensor([0.17,0.235,0.33,-0.05,2.3,-0.05,0.055,0.04,0.04]))
    
    # prior = utils.BoxUniform(low=torch.Tensor(parameters_all_ERP[participants[ind_part]])-torch.Tensor([0.02,0.02,0.02,0.2,0.2,0.2,0.01,0.01,0.01]),
    #                          high=torch.Tensor(parameters_all_ERP[participants[ind_part]])+torch.Tensor([0.02,0.02,0.02,np.min([0.2, -parameters_all_ERP[participants[ind_part]][3]]),0.2,np.min([0.2, -parameters_all_ERP[participants[ind_part]][5]]),0.01,0.01,0.01]))
    
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(lambda x: simulator_model(x,0,[detected_code,detected_code]), prior, prior_returns_numpy)
    # simulator_wrapper = process_simulator(lambda x: simulator_model(x,None,codes), prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)
    num_simulations = 15_000
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=num_simulations)
    print(x.shape)
    # theta = torch.repeat_interleave(theta,x.shape[1],dim=0)
    x = torch.reshape(x,(-1,x.shape[-1]))
    # instantiate the neural density estimator
    neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net)
    # setup the inference procedure with NPE
    inferer = NPE(prior=prior, density_estimator=neural_posterior)
    density_estimator = inferer.append_simulations(theta, x).train(training_batch_size=256,stop_after_epochs=35,max_num_epochs=300)
    posterior = inferer.build_posterior(density_estimator)
    # torch.save(posterior,"D:/s.velut/Documents/These/DCC/SBI/overall_posterior"+str(ind_part)+".pkl")
    # torch.save(inferer,"D:/s.velut/Documents/These/DCC/SBI/overall_inferer"+str(ind_part)+".pkl")

    ############################ Code detection with SBI ##########################
    # Create the simulated VEP templates 
    y_pred_all = []
    y_pred_all2 = []
    Xsim = []
    ysim = []
    y_true = trial_labels-1
    # X_detected_code = trials[ind_part,n_class*n_cal,0:1,:888]/np.linalg.norm(trials[ind_part,n_class*n_cal,0:1,:888])
    # X = trials[ind_part,:,0:1,:888]/np.linalg.norm(trials[ind_part,:,0:1,:888])
    # codes = np.array([codesdict[i] for i in range(len(codesdict))],dtype=np.uint8)
    # xemp = X_detected_code
    # sl = posterior.set_default_x(xemp).sample((1000,))
    # hist = [np.histogram(sl[:,i], bins=500) for i in range(sl.shape[1])]
    # ind_max = [np.argmax(hist[i][0]) for i in range(len(hist))]
    # theta_obs = torch.Tensor([hist[i][1][ind_max[i]] for i in range(len(ind_max))])
    # for i in range(X.shape[0]):
        
    #     # theta_obs = torch.as_tensor([0.157,0.2, 0.29, 0.000025, -0.00005, 0.000001, 0.01, 0.03, 0.01])

    #     Xsim.append([simulator_model(theta_obs,only_template=True,code=j, codes=codes).numpy() for j in range(5)])
    #     ysim.append(np.linspace(0,5-1,5).astype(np.uint8))
    # Xsim = np.concatenate(Xsim,axis=0)
    # ysim = np.concatenate(ysim,axis=0)

    # accuracy = cross_validate(X, np.expand_dims(Xsim,1), y_true, codes, y_syn=ysim,n_folds=5)
    # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/overall_CV_accuracy_participant_RNN_"+str(ind_part)+".npy",np.mean(accuracy,axis=1))
    # print("Emp>emp:", accuracy[0, :].mean())
    # print("Emp>syn:", accuracy[1, :].mean())
    # print("Syn>emp:", accuracy[2, :].mean())
    # print("Syn>syn:", accuracy[3, :].mean())

    # # # # # # # # # # # # # # # # # # # # # # # # # # Other way
    # y_pred_all = []
    # theta_all = []
    nfold = 1
    # print(y_true)
    # if participants[ind_part]=="P2":
    #     X = trials[:,1:2,:]#/np.linalg.norm(trials[ind_part,:,1:2,:])
    # else: 
    #     X = trials[:,0:1,:]#/np.linalg.norm(trials[ind_part,:,0:1,:])
        
    # # X = torch.Tensor([np.array(compute_summary_statistics(xk)) for xk in X])

    # # y_true = mat1["y"][:,0]
    # # X = mat1["X"][4].T[:,:ntimes]/np.linalg.norm(mat1["X"][4].T[:,:ntimes])
    # # codes = mat['V'].T[:,:]  #132 if X is 1056 shape
    # for i in range(X.shape[0]):
    #     y_pred = []
    #     corr = []
    #     for n in range(nfold):
    #         xemp = X[i]
    #         sl = posterior.set_default_x(xemp).sample((1000,))
    #         hist = [np.histogram(sl[:,i], bins=500) for i in range(sl.shape[1])]
    #         ind_max = [np.argmax(hist[i][0]) for i in range(len(hist))]
    #         theta_obs = torch.Tensor([hist[i][1][ind_max[i]] for i in range(len(ind_max))])
    #         y_pred.append(np.argmax([correlation(xemp,simulator_model(theta_obs,code=j,only_template=True, codes=codes).numpy()) for j in range(5)]))
    #         # y_pred.append(np.argmax([correlation(xemp,np.abs(simulator_model(theta_obs,code=j,only_template=True, codes=codes).numpy()[:888])) for j in range(5)]))
    #     y_pred_all.append(max(y_pred,key=y_pred.count))
    #     theta_all.append(theta_obs.numpy())
    # print(np.mean(y_pred_all==y_true))
    # # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/overall_accuracy_participant_RNN_Inv_AllCode"+str(ind_part)+".npy",np.array([np.mean(np.array(y_pred_all)==y_true)]))
    # # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/theta_RNN_Inv_AllCode"+str(ind_part)+".npy",np.array(theta_obs))

    # Other way 2 
    # y_pred_all = []
    # nfold = 1
    # y_true = y
    # X = trials[ind_part,:,0:1,:888]/np.linalg.norm(trials[ind_part,:,0:1,:888])
    # # X = torch.Tensor([np.array(compute_summary_statistics(xk)) for xk in X])
    # codes = np.array([codesdict[i] for i in range(len(codesdict))],dtype=np.uint8)

    # # y_true = mat1["y"][:,0]
    # # X = mat1["X"][4].T[:,:ntimes]/np.linalg.norm(mat1["X"][4].T[:,:ntimes])
    # # codes = mat['V'].T[:,:]  #132 if X is 1056 shape
    # X_detected_code = trials[ind_part,n_class*n_cal,0:1,:888]/np.linalg.norm(trials[ind_part,n_class*n_cal,0:1,:888])
    # sl = posterior.set_default_x(X_detected_code).sample((1000,))
    # hist = [np.histogram(sl[:,i], bins=500) for i in range(sl.shape[1])]
    # ind_max = [np.argmax(hist[i][0]) for i in range(len(hist))]
    # theta_obs = torch.Tensor([hist[i][1][ind_max[i]] for i in range(len(ind_max))])
    # for i in range(X.shape[0]):
    #     y_pred = []
    #     corr = []
    #     xemp = X[i]
    #     for n in range(nfold):
    #         y_pred.append(np.argmax([correlation(xemp,np.abs(simulator_model(theta_obs,code=j,only_template=True, codes=codes).numpy()[:888])) for j in range(5)]))
    #     y_pred_all.append(max(y_pred,key=y_pred.count))
    # print(np.mean(y_pred_all==y_true))
    # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/overall_accuracy_2_participant_RNN_InvBiggestAmp"+str(ind_part)+".npy",np.array([np.mean(np.array(y_pred_all)==y_true)]))

    # Simulation to simulation
    # # # # prior = utils.BoxUniform(low=torch.FloatTensor([0.13,0.19,0.28,-0.5,0.3,-0.1,0.008,0.029,0.014]), high=torch.FloatTensor([0.14,0.2,0.29,-0.4,0.4,0,0.009,0.03,0.015]))
    # # # # prior, num_parameters, prior_returns_numpy = process_prior(prior)
    X_sim = []
    Y_sim = []
    theta_sim = []
    for j in range(5):
        simulator_wrapper = process_simulator(lambda x: simulator_model(x,j,codes), prior, prior_returns_numpy)
        check_sbi_inputs(simulator_wrapper, prior)
        num_simulations = 50
        theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=num_simulations)
        x = torch.reshape(x,(-1,x.shape[-1]))
        X_sim.append(x)
        Y_sim.append([j]*num_simulations)
        theta_sim.append(theta)

    X_sim = np.reshape(X_sim,(-1,np.array(X_sim).shape[-1]))
    Y_sim = np.reshape(Y_sim,-1)
    theta_sim = np.reshape(theta_sim,(-1,9))
    print(X_sim.shape,Y_sim.shape)
    y_pred_all = []
    theta_all = []
    rmse_all = []
    theta  

    for i in range(X_sim.shape[0]):
        y_pred = []
        corr = []
        for n in range(nfold):
            xemp = X_sim[i][:1056]
            sl = posterior.set_default_x(xemp).sample((1000,))
            hist = [np.histogram(sl[:,i], bins=500) for i in range(sl.shape[1])]
            ind_max = [np.argmax(hist[i][0]) for i in range(len(hist))]
            theta_obs = torch.Tensor([hist[i][1][ind_max[i]] for i in range(len(ind_max))])
            y_pred.append(np.argmax([correlation(xemp,simulator_model(theta_obs,code=j,only_template=True, codes=codes).numpy()[:1056]) for j in range(5)]))
        y_pred_all.append(max(y_pred,key=y_pred.count))
        theta_all.append(theta_obs.numpy())
    rmse_all = np.sqrt(np.mean((theta_obs-theta_sim).numpy()**2,axis=0))
    print(rmse_all.shape, rmse_all)
    print(Y_sim,np.array(y_pred_all))
    print(np.mean(y_pred_all==Y_sim))
    # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/overall_accuracy_participant_RNN_Sim2Sim_Inv_AllCode"+str(ind_part)+".npy",np.array([np.mean(y_pred_all==Y_sim)]))
    # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/theta_RNN_Sim2Sim_obs_Inv_AllCode"+str(ind_part)+".npy",np.array(theta_all))
    # np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/theta_RNN_Sim2Sim_true_Inv_AllCode"+str(ind_part)+".npy",np.array(theta_sim))
    np.save("D:/s.velut/Documents/These/DCC/SBI/results/overall_sbi/RMSE_RNN_Sim2Sim_true_Inv"+str(ind_part)+".npy",rmse_all)
