import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbi import analysis, utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyntbci.utilities import correlation
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets import posterior_nn

import pyntbci
import scipy.io

from simulated_cvep import forward_model

mat = scipy.io.loadmat('D:/s.velut/Downloads/DSC_2018.00122_448_v3/derivatives/offline/sub-01/sub-01_mat.mat')
codes = mat['V'].T

seed = 0
torch.manual_seed(seed);


def simulator_model(theta,code = None, only_template=False):
    fs = 180
    pr = 60
    length_vep=0.3
    V = codes
    
    V = torch.repeat_interleave(torch.tensor(V), int(fs / pr))
    # V = torch.repeat_interleave(torch.tensor(np.tile(V,5)), int(fs / pr))
    t = torch.arange(0, length_vep, 1 / fs)

    delays = theta[:3].unsqueeze(1)  # shape [n_peaks, 1]
    amplitudes = theta[3:6].unsqueeze(1)  # shape [n_peaks, 1]
    sigmas = theta[6:9].unsqueeze(1)  # shape [n_peaks, 1]

    # Add extra dimensions for torch, shape [1, t_vep]
    t = t.unsqueeze(0)  

    # Compute gaussians, shape [n_peaks, t_vep]
    vep_components = forward_model.gaussian(t, delays, amplitudes, sigmas)

    # Sum over components, shape [t_vep]
    vep = vep_components.sum(dim=0)  

    #Convert to float for torch
    V = V.float()
    vep = vep.float()

    #Add extra dimensions for torch
    V = V.view(20, 1, -1)  # shape [1, 1, t_code]
    vep = vep.view(1, 1, -1) # shape [1, 1, t_vep]

    X = torch.nn.functional.conv1d(V, vep, padding='same') # Convolve V with vep

    V = V.squeeze()
    sim_signal = X.squeeze() # shape [t_code]
    sim_vep = X.squeeze() # shape [t_code]

    noise=0
    snr = 0.08
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
    

def train(parameters, posterior_path, inferer_path):
    embedding_net = CNNEmbedding(
    input_shape=(1, 1134),
        in_channels=1,
        out_channels_per_layer=[6,12,24],
        num_conv_layers=3,
        num_linear_layers=1,
        output_dim=12,
        kernel_size=1,
        pool_kernel_size=3
    )

    # set prior distribution for the parameters
    prior = utils.BoxUniform(low=torch.FloatTensor([0.050,0.050,0.050,-10,-10,-10,0.005,0.005,0.005]), high=torch.FloatTensor([0.500,0.500,0.500,10,10,10,0.05,0.05,0.05]))

    # make a SBI-wrapper on the simulator object for compatibility
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator_model, prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)

    # instantiate the neural density estimator
    neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net)

    # setup the inference procedure with NPE
    inferer = NPE(prior=prior, density_estimator=neural_posterior)

    # run the inference procedure on one round and 10000 simulated data points
    num_simulations = 5_000
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=num_simulations)
    x = torch.reshape(x,(-1,x.shape[-1]))
    theta = torch.repeat_interleave(theta,20,dim=0)

    density_estimator = inferer.append_simulations(theta, x).train(training_batch_size=256,stop_after_epochs=25,max_num_epochs=500)
    posterior = inferer.build_posterior(density_estimator)

    torch.save(posterior,posterior_path)
    torch.save(inferer,inferer_path)

    return 0

def sample():

    return 0

def correlate():

    return 0