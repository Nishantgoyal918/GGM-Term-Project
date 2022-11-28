from dataset import getSets
from BNNModel import BayesianMnistNet

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb

import os

import argparse as args

import easydict

import math
import scipy.stats as stats


args = {
    'filteredclass': 5,
    'testclass': 4,
    'savedir': None,
    'notrain': False,
    'nepochs': 10,
    'nbatch': 64,
    'nruntests': 50,
    'learningrate': 5e-3,
    'numnetworks': 1 # No. of models to train
    }



def saveModels(models, savedir) :
    
    for i, m in enumerate(models) :
        
        saveFileName = os.path.join(savedir, "model{}.pth".format(i))
        
        torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))
    
def loadModels(savedir) :
    
    models = []
    
    for f in os.listdir(savedir) :
        
        model = BayesianMnistNet(p_mc_dropout=None)		
        model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
        models.append(model)
        
    return models

if __name__ == "__main__" :
    
    # parser = args.ArgumentParser(description='Train a BNN on Mnist')

    # parser.add_argument('--filteredclass', type=int, default = 5, choices = [x for x in range(10)], help="The class to ignore during training")
    # parser.add_argument('--testclass', type=int, default = 4, choices = [x for x in range(10)], help="The class to test against that is not the filtered class")


    # parser.add_argument('--savedir', default = None, help="Directory where the models can be saved or loaded from")
    # parser.add_argument('--notrain', action = "store_true", help="Load the models directly instead of training")

    # parser.add_argument('--nepochs', type=int, default = 10, help="The number of epochs to train for")
    # parser.add_argument('--nbatch', type=int, default = 64, help="Batch size used for training")
    # parser.add_argument('--nruntests', type=int, default = 50, help="The number of pass to use at test time for monte-carlo uncertainty estimation")
    # parser.add_argument('--learningrate', type=float, default = 5e-3, help="The learning rate of the optimizer")
    # parser.add_argument('--numnetworks', type=int, default = 10, help="The number of networks to train to make an ensemble")

    args = easydict.EasyDict(args)

    # args = parser.parse_args()
    plt.rcParams["font.family"] = "serif"


    train, test = getSets(filteredClass = args.filteredclass)
    train_filtered, test_filtered = getSets(filteredClass = args.filteredclass, removeFiltered = False)

    N = len(train)

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.nbatch)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.nbatch)

    batchLen = len(train_loader)
    digitsBatchLen = len(str(batchLen))

    models = []

    # Training or Loading
    if args.notrain :
        
        models = loadModels(args.savedir)
        
    else :

        for i in np.arange(args.numnetworks) :
            print("Training model {}/{}:".format(i+1, args.numnetworks))
            
            #Initialize the model
            model = BayesianMnistNet(p_mc_dropout=None) #p_mc_dropout=None will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
            loss = torch.nn.NLLLoss(reduction='mean') #negative log likelihood will be part of the ELBO
            
            optimizer = Adam(model.parameters(), lr=args.learningrate)
            optimizer.zero_grad()
            
            for n in np.arange(args.nepochs) :

                w_prob = 0.0
                w_modelloss = 0.0
                w_loss = 0.0
                
                for batch_id, sampl in enumerate(train_loader) :
                    
                    images, labels = sampl
                    
                    pred = model(images, stochastic=True)
                    
                    logprob = loss(pred, labels)
                    l = N*logprob
                    
                    modelloss = model.evalAllLosses()
                    l += modelloss
                    
                    optimizer.zero_grad()
                    l.backward()
                    
                    optimizer.step()
                    w_prob += torch.exp(-logprob.detach().cpu()).item()
                    w_modelloss += modelloss.detach().cpu().item()
                    w_loss += l.detach().cpu().item()
                    
                    print("\r", ("\tEpoch {}/{}: Train step {"+(":0{}d".format(digitsBatchLen))+"}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
                                                                                                    n+1, args.nepochs,
                                                                                                    batch_id+1,
                                                                                                    batchLen,
                                                                                                    torch.exp(-logprob.detach().cpu()).item(),
                                                                                                    modelloss.detach().cpu().item(),
                                                                                                    l.detach().cpu().item()), end="")
                w_prob = w_prob / len(train_loader)
                w_modelloss = w_modelloss / len(train_loader)
                w_loss = w_loss / len(train_loader)

                wandb.log({
                        'probability': w_prob,
                        'model loss': w_modelloss,
                        'total loss': w_loss
                })
            print("")
            
            models.append(model)

    if args.savedir is not None :
        saveModels(models, args.savedir)
    

    means_ = []
    std_ = []

    for i in range(10):
        testclass = i
        if testclass != args.filteredclass :
            train_filtered_seen, test_filtered_seen = getSets(filteredClass = testclass, removeFiltered = False)
            # train_filtered_seen, test_filtered_seen = getSets(filteredClass = 9, removeFiltered = False)

            print("")
            print(f"Testing against seen class {testclass}:")
            
            with torch.no_grad() :
            
                samples = torch.zeros((args.nruntests, len(test_filtered_seen), 10))
                
                test_loader = DataLoader(test_filtered_seen, batch_size=len(test_filtered_seen))
                images, labels = next(iter(test_loader))
                
                for i in np.arange(args.nruntests) :
                    print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
                    model = np.random.randint(args.numnetworks)
                    model = models[model]
                    
                    samples[i,:,:] = torch.exp(model(images))
                
                print("")
                
                withinSampleMean = torch.mean(samples, dim=0)
                samplesMean = torch.mean(samples, dim=(0,1))
                
                withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
                acrossSamplesStd = torch.std(withinSampleMean, dim=0)

                # print(withinSampleMean)
                # print(samplesMean)
                
                print("")
                print("Class prediction analysis:")
                print("\tMean class probabilities:")
                print(samplesMean)

                means_.append(samplesMean.cpu().detach().numpy())

                print("\tPrediction standard deviation per sample:")
                print(withinSampleStd)

                std_.append(withinSampleStd.cpu().detach().numpy())

                print("\tPrediction standard deviation across samples:")
                print(acrossSamplesStd)
        
        else:
            print("")
            print("Testing against unseen class:")

            with torch.no_grad() :

                samples = torch.zeros((args.nruntests, len(test_filtered), 10))
                
                test_loader = DataLoader(test_filtered, batch_size=len(test_filtered))
                images, labels = next(iter(test_loader))
                
                for i in np.arange(args.nruntests) :
                    print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
                    model = np.random.randint(args.numnetworks)
                    model = models[model]
                    
                    samples[i,:,:] = torch.exp(model(images))

            print("")
            
            withinSampleMean = torch.mean(samples, dim=0)
            samplesMean = torch.mean(samples, dim=(0,1))
            
            withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
            acrossSamplesStd = torch.std(withinSampleMean, dim=0)
            
            print("")
            print("Class prediction analysis:")
            print("\tMean class probabilities:")
            print(samplesMean)

            means_.append(samplesMean.cpu().detach().numpy())

            print("\tPrediction standard deviation per sample:")
            print(withinSampleStd)

            std_.append(withinSampleStd.cpu().detach().numpy())
            print("\tPrediction standard deviation across samples:")
            print(acrossSamplesStd)
    
    means_ = np.vstack(means_)
    std_ = np.vstack(std_)


    pure_white_means_ = None
    pure_white_std = None
    print("")
    print("Testing against pure white noise:")

    with torch.no_grad() :

        l = 1000
        
        samples = torch.zeros((args.nruntests, l, 10))
        
        random = torch.rand((l,1,28,28))
        
        for i in np.arange(args.nruntests) :
            print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
            model = np.random.randint(args.numnetworks)
            model = models[model]
            
            samples[i,:,:] = torch.exp(model(random))

    print("")
    
    withinSampleMean = torch.mean(samples, dim=0)
    samplesMean = torch.mean(samples, dim=(0,1))
    
    withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
    acrossSamplesStd = torch.std(withinSampleMean, dim=0)
    
    print("")
    print("Class prediction analysis:")
    print("\tMean class probabilities:")
    print(samplesMean)

    pure_white_means_ = samplesMean.cpu().detach().numpy()
    print("\tPrediction standard deviation per sample:")
    print(withinSampleStd)

    pure_white_std = withinSampleStd.cpu().detach().numpy()

    print("\tPrediction standard deviation across samples:")
    print(acrossSamplesStd)


    # Plots
    # Heatmap
    ylabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    xlabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    im = ax.imshow(means_, cmap="Blues")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_title("probability heatmap for different classes")
    plt.show()


    # Plot for seen data (digit 9)
    for i in range(len(means_[9])):
        mu = means_[9][i]
        # variance = 1
        sigma = std_[9][i] #math.sqrt(variance)
        x = np.linspace(-0.5, 1.5, 500)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label=f"digit {i}")
        plt.legend()
        plt.title("(a) Probability distribution when predicting for seen class [digit 9]")
        plt.xlabel("Probability")
        plt.show()
    

    # Plot for unseen data (digit 5)
    for i in range(len(means_[5])):
        mu = means_[5][i]
        # variance = 1
        sigma = std_[5][i] #math.sqrt(variance)
        x = np.linspace(-0.5, 1.5, 500)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label=f"digit {i}")
        plt.legend()
        plt.title("(a) Probability distribution when predicting for unseen class [digit 5]")
        plt.xlabel("Probability")
        plt.show()
    

    # Plot for white noise
    for i in range(len(pure_white_means_)):
        mu = pure_white_means_[i]
        # variance = 1
        sigma = pure_white_std[i] #math.sqrt(variance)
        x = np.linspace(-0.5, 1.5, 500)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label=f"digit {i}")
        plt.legend()
        plt.title("(a) Probability distribution when predicting for white noise")
        plt.xlabel("Probability")
        plt.show()