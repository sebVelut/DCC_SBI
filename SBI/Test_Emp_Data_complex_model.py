import argparse
import yaml
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
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

seed = 0
torch.manual_seed(seed);

def correlation(
        A,
        B,
):
    """Compute the correlation coefficient. Computed between two sets of variables.

    Parameters
    ----------
    A: NDArray
        The first set of variables of shape (n_A, n_samples).
    B: NDArray
        The second set of variables of shape (n_B, n_samples).

    Returns
    -------
    scores: NDArray
        The correlation matrix of shape (n_A, n_B).
    """
    if A.ndim == 1:
        A = A[np.newaxis, :]
    if B.ndim == 1:
        B = B[np.newaxis, :]
    assert A.shape[1] == B.shape[1], f"Number of samples in A ({A.shape[1]}) does not equal B ({B.shape[1]})"

    # Demean
    Az = A - np.mean(A, axis=1, keepdims=True)
    Bz = B - np.mean(B, axis=1, keepdims=True)

    # Sum of squares
    ssA = np.sum(Az ** 2, axis=1, keepdims=True)
    ssB = np.sum(Bz ** 2, axis=1, keepdims=True)

    # Correlation coefficient
    outer = np.dot(ssA, ssB.T)
    scores = np.dot(Az, Bz.T) / np.sqrt(outer)

    return scores


def simulator_model(theta,code = None, only_template=False):
    fs = 180
    pr = 60
    length_vep=0.3
    V = codes.copy()
    
    # V = torch.repeat_interleave(torch.tensor(V), int(fs / pr),dim=1)
    V = torch.repeat_interleave(torch.tensor(np.tile(V,5)), int(fs / pr),dim=1)
    V = torch.Tensor(np.concatenate([V[:,0:1],np.diff(V)],axis=1)%255)
    # print(V[0,:40])
    t = torch.arange(0, length_vep, 1 / fs)

    sim_vep, sim_signal = forward_model.fwd(t, V, theta, 0)

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
    

def train(parameters):
    embedding_net = CNNEmbedding(
    input_shape=(1, 5670),
        in_channels=1,
        out_channels_per_layer=parameters["out_channels"],
        num_conv_layers=len(parameters["out_channels"]),
        num_linear_layers=1,
        output_dim=parameters["output_dim"],
        kernel_size=1,
        pool_kernel_size=3
    )

    # set prior distribution for the parameters
    prior = utils.BoxUniform(low=torch.FloatTensor(parameters["low"]), high=torch.FloatTensor(parameters["high"]))

    # make a SBI-wrapper on the simulator object for compatibility
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator_model, prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)

    # instantiate the neural density estimator
    neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net)

    # setup the inference procedure with NPE
    inferer = NPE(prior=prior, density_estimator=neural_posterior)

    # run the inference procedure on one round and 10000 simulated data points
    num_simulations = parameters["num_simulations"]
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=num_simulations)
    x = torch.reshape(x,(-1,x.shape[-1]))
    theta = torch.repeat_interleave(theta,20,dim=0)

    density_estimator = inferer.append_simulations(theta, x).train(training_batch_size=256,stop_after_epochs=25,max_num_epochs=500)
    posterior = inferer.build_posterior(density_estimator)

    torch.save(posterior,parameters["posterior_path"])
    torch.save(inferer,parameters["inferer_path"])

    return posterior, inferer

def sample(posterior_path,x_obs,num_simulations):
    posterior = torch.load(posterior_path)
    samples = posterior.set_default_x(x_obs).sample((num_simulations,))

    return samples

def correlate_one(posterior_path,x_obs,num_simulations):
    posterior = torch.load(posterior_path)
    sl = posterior.set_default_x(x_obs).sample((num_simulations,)).tolist()
    theta_obs = torch.tensor(max(sl,key=sl.count))
    t_a = []

    for i in range(20):
        template_pred = simulator_model(theta_obs,only_template=True,code=i).numpy()
        t_a.append(template_pred)
        # print(f"template {i+1} avec correlation {np.corrcoef(np.mean(x_pred,axis=0),template_pred[i])[0,1]}")
        print(f"template {i+1} avec correlation {correlation(x_obs,template_pred)}")

    ind_max = np.argmax([correlation(x_obs,t_a[i]) for i in range(20)])    
    return ind_max


def score_correlation(X,Y,posterior_path,num_samples):
    """
    Compute the score of the correlation between the predicted and observed signals.

    Parameters:
    ----------
    X : NDArray
        The observed signals of shape (n_samples, n_time).
    Y : NDArray
        The labels of the observed signals of shape (n_samples,).
    posterior_path : str
        The path to the posterior model.
    num_simulations : int
        The number of simulations to run.
    """
    y_pred = []

    for i in range(X.shape[0]):
        x_obs = X[i]
        y_obs = Y[i]
        # print(f"X {x_obs} et Y {y_obs}")
        # print(f"X {x_obs} et Y {y_obs}")
        ind_max = correlate_one(posterior_path,x_obs,num_samples)
        print(f"template {ind_max+1} avec correlation {correlation(x_obs,y_obs)}")
        y_pred.append(ind_max)

    return np.mean(y_pred==Y)

def fold_rcca(parameters,data_path,rcca_path):
    assert rcca_path is not None, "Please provide a path to save the rcca plot"
    posterior = torch.load(parameters["posterior_path"])
    mat = scipy.io.loadmat(data_path)
    X = mat["X"][4].T
    Y = mat["y"].flatten()
    sl = posterior.set_default_x(X).sample((parameters["num_samples"],))

    X = np.array([simulator_model(s) for s in sl]).reshape(-1,1,5670)
    y = np.tile(np.arange(0,20,1),1000)
    paper_codes = mat['V'].T
    V = np.concatenate([paper_codes[:,0:1],np.diff(paper_codes)],axis=1)%255
    fs = int(mat["fs"])
    fr = 60
    n_trials, n_channels, n_samples = X.shape
    n_classes = V.shape[0]

    trialtime = 6.2  # limit trials to a certain duration in seconds
    intertrialtime = 1.0  # ITI in seconds for computing ITR
    n_samples = int(trialtime * fs)

    # Chronological cross-validation
    n_folds = 5
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))
    accuracy = np.zeros(n_folds)
    for i_fold in range(n_folds):
        # Split data to train and test set
        X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
        X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

        # Train template-matching classifier
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="rise", encoding_length=0.3, onset_event=True)
        rcca.fit(X_trn, y_trn)

        # Apply template-matching classifier
        yh_tst = rcca.predict(X_tst)

        # Compute accuracy
        accuracy[i_fold] = np.mean(yh_tst == y_tst)

    # Compute ITR
    itr = pyntbci.utilities.itr(n_classes, accuracy, trialtime + intertrialtime)

    # Plot accuracy (over folds)
    plt.figure(figsize=(15, 3))
    plt.bar(np.arange(n_folds), accuracy)
    plt.axhline(accuracy.mean(), linestyle='--', alpha=0.5, label="average")
    plt.axhline(1 / n_classes, color="k", linestyle="--", alpha=0.5, label="chance")
    plt.xlabel("(test) fold")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Chronological cross-validation")
    plt.tight_layout()

    # Print accuracy (average and standard deviation over folds)
    print(f"Accuracy: avg={accuracy.mean():.2f} with std={accuracy.std():.2f}")
    print(f"ITR: avg={itr.mean():.1f} with std={itr.std():.2f}")
    plt.savefig(rcca_path)

def main(parameters_file, run_type, data_path, scores_path,retrain=False, plot_graph=False,rcca_path=None):

    with open("configuration.yaml", "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if run_type == "train":
        if retrain==True or not os.path.exists(parameters["posterior_path"]):
            posterior, inferer = train(parameters)
    elif run_type == "score":
        mat = scipy.io.loadmat(data_path)
        X = mat["X"][4].T
        Y = mat["y"].flatten()
        scores = score_correlation(X,Y,parameters["posterior_path"],parameters["num_samples"])

        np.save(scores_path+f"/score_SBI_{datetime.datetime.today().strftime('%m-%d-%H_%M_%S')}",scores)
    elif run_type=="all":
        if retrain==True or not os.path.exists(parameters["posterior_path"]):
            posterior, inferer = train(parameters)
        mat = scipy.io.loadmat(data_path)
        X = mat["X"][4].T
        Y = mat["y"].flatten()
        scores = score_correlation(X,Y,parameters["posterior_path"],parameters["num_samples"])
        np.save(scores_path+f"/score_SBI_{datetime.datetime.today().strftime('%m-%d-%H_%M_%S')}",scores)
    elif run_type == "rcca":
        fold_rcca(parameters,data_path,rcca_path)

    if plot_graph:
        true_parameter = torch.as_tensor([0.095,0.115, 0.160, -4, 6, -3, 0.015, 0.020, 0.018])
        x_observed = simulator_model(true_parameter,code=10)
        samples = torch.load(parameters["posterior_path"]).set_default_x(x_observed).sample((parameters["num_samples"],))
        low,high = parameters["low"],parameters["high"]
        limits = [[low[i], high[i]] for i in range(len(low))]
        # create the figure
        fig, ax = analysis.pairplot(
            samples,
            points=true_parameter,
            labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$", r"$\theta_6$",r"$\theta_7$", r"$\theta_8$", r"$\theta_9$"],
            limits=limits,
            fig_kwargs=dict(
                points_colors="r",
                points_offdiag={"markersize": 6},
            ),
            figsize=(5, 5),
        )

    return scores


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters_file", help="The file with all the parameters that will describe the vairables useful for the pipeline. Need to be a .json file")
    parser.add_argument("--data_path", help="The path to the data file")
    parser.add_argument("--scores_path", help="The path to store the score file")
    parser.add_argument("--run_type", default="all", help="The type of run to do. Can be train, score or all")
    parser.add_argument("--retrain", default=False, type=bool, help="Boolean to retrain the model or not")
    parser.add_argument("--plot_graph", default=False, type=bool, help="Boolean to plot the graph or not")
    parser.add_argument("--rcca_path", default=None, help="The path to the data file")

    args = parser.parse_args()

    mat = scipy.io.loadmat(args.data_path)
    codes = mat['V'].T


    main(args.parameters_file, args.run_type, args.data_path, args.scores_path, args.retrain, args.plot_graph)