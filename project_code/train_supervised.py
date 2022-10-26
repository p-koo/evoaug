#!/usr/bin/env python

## ---------- Import packages ----------
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


import os, h5py, gc, math
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# EvoAug scripts--check relative paths
from models import *
from zoo import *
from augmentations import aug_pad_end
from datamodules import *
from utils import *


## ---------- Parse arguments ----------
parser = argparse.ArgumentParser(description="Train supervised model", 
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("data_path", metavar="/PATH/TO/DATA/", type=str,
                    help="path to HDF5 file containing data, pre-split into training, validation, and test sets")

parser.add_argument("-loss", metavar="LOSS_FUNCTION", type=str, 
                    help="loss function to use in training model; possible loss functions are \"BCE\" (binary cross-entropy, for classification tasks) and \"MSE\" (mean squared error, for regression tasks)", 
                    default="BCE")

parser.add_argument("-expt_name", metavar="EXPERIMENT_NAME", type=str, 
                    help="name of experiment that will be used as file prefix", 
                    default="Expt")
parser.add_argument("-output_dir", metavar="/PATH/TO/OUTPUT/", type=str, 
                    help="directory where output files will be stored; experiment name will be appended to results files", 
                    default="./Expt/")

parser.add_argument("-model_type", metavar="MODEL_TYPE", type=str,
                    help="model architecture type; possible model types are \"CNN_S\", \"Basset\", \"DeepSTARR\"", 
                    default="Basset")
parser.add_argument("-d", metavar="NUM_FILTERS", type=int,
                    help="number of first layer convolutional filters to use", 
                    default=300)
parser.add_argument("-S", metavar="MAXPOOL_SIZE", type=int,
                    help="first max-pooling layer kernel size in encoder; only for use with option \"-model_type\" set to \"CNN_S\"", 
                    default=4)

parser.add_argument("-lr", metavar="LEARNING_RATE", type=float,
                    help="learning rate for Adam optimizer", 
                    default=1e-3)
parser.add_argument("-wd", metavar="WEIGHT_DECAY", type=float,
                    help="weight decay (L2 penalty) for Adam optimizer", 
                    default=1e-6)
parser.add_argument("-patience", metavar="PATIENCE", type=int,
                    help="patience used in early stopping", 
                    default=10)

parser.add_argument("-augs", metavar="AUGMENTATION_STRING", type=str,
                    help="string specifying augmentations to use, comma delimited; possible augmentations are \"invert\", \"delete\", \"translocate\", \"insert\", \"rc\", \"mutate\", \"noise_gauss\" (e.g., \"noise_gauss,rc,translocate,insert\")", 
                    default="")
parser.add_argument("-sample_augs_num", metavar="SAMPLE_AUGS_NUM", type=int,
                    help="(maximum) number of augmentations to sample from AUGMENTATION_STRING denoted in option \"-augs\" and apply to each sequence during training; default value of None results in the total number of augmentations specified in AUGMENTATION_STRING", 
                    default=None)
parser.add_argument("-sample_augs_hard", metavar="FLAG_SAMPLE_AUGS_HARD", type=bool,
                    help="boolean denoting whether to sample exactly SAMPLE_AUGS_NUM augmentations from AUGMENTATION_STRING to apply to each sequence during training (True) or instead to sample between 1 and SAMPLE_AUGS_NUM augmentations from AUGMENTATION_STRING to apply to each sequence during training (False)", 
                    default=True)

parser.add_argument("-invert_min", metavar="INVERT_MIN", type=int,
                    help="in inversion augmentation, minimum length of inversion", 
                    default=0)
parser.add_argument("-invert_max", metavar="INVERT_MAX", type=int,
                    help="in inversion augmentation, maximum length of inversion", 
                    default=30)
parser.add_argument("-delete_min", metavar="DELETE_MIN", type=int,
                    help="in deletion augmentation, minimum length of deletion", 
                    default=0)
parser.add_argument("-delete_max", metavar="DELETE_MAX", type=int,
                    help="in deletion augmentation, maximum length of deletion", 
                    default=30)
parser.add_argument("-shift_min", metavar="SHIFT_MIN", type=int,
                    help="in translocation augmentation, minimum number of places by which position can be shifted", 
                    default=0)
parser.add_argument("-shift_max", metavar="SHIFT_MAX", type=int,
                    help="in translocation augmentation, maximum number of places by which position can be shifted", 
                    default=30)
parser.add_argument("-insert_min", metavar="INSERT_MIN", type=int,
                    help="in insertion augmentation, minimum length of insertion", 
                    default=0)
parser.add_argument("-insert_max", metavar="INSERT_MAX", type=int,
                    help="in insertion augmentation, maximum length of insertion", 
                    default=30)
parser.add_argument("-rc_prob", metavar="RC_PROB", type=float,
                    help="in reverse complement augmentation, probability for each sequence to be \"mutated\" to its reverse complement", 
                    default=0.5)
parser.add_argument("-mutate_frac", metavar="MUTATE_FRAC", type=float,
                    help="in random mutation augmentation, fraction of each sequence's nucleotides to mutate", 
                    default=0.1)
parser.add_argument("-noise_mean", metavar="NOISE_MEAN", type=float,
                    help="in Gaussian noise addition augmentation, mean of Gaussian distribution from which noise is drawn", 
                    default=0.0)
parser.add_argument("-noise_std", metavar="NOISE_STDEV", type=float,
                    help="in Gaussian noise addition augmentation, standard deviation of Gaussian distribution from which noise is drawn", 
                    default=0.2)

parser.add_argument("-batch_size", metavar="BATCH_SIZE", type=int, 
                    help="batch size used in training", 
                    default=128)
parser.add_argument("-epochs", metavar="MAX_EPOCHS", type=int, 
                    help="number of epochs to train model", 
                    default=100)

parser.add_argument("-alphabet", metavar="ALPHABET", type=str, 
                    help="order of nucleotide channels in one-hot encoding of sequences x (e.g., \"ATCG\"), if custom", 
                    default="ACGT")
parser.add_argument("-filter_viz_subset", metavar="SIZE", type=int,
                    help="size of subset of test set to use for filter visualization; default value of None results in usage of whole test set", 
                    default=None)


# Parse args
args = parser.parse_args()

expt_name = args.expt_name
output_dir = args.output_dir
make_directory(output_dir) # if necessary, make directory in which to save results

aug_string = args.augs.lower()
augs = aug_string.split(",")
set_augs = ["invert", "delete", "translocate", "insert", "rc", "mutate", "noise_gauss"]
if args.augs != "":
    assert all(aug in set_augs for aug in augs), "unrecognized augmentation in user-defined AUGMENTATION_STRING"
    if args.sample_augs_num is not None:
        assert 0 < args.sample_augs_num <= len(augs), "cannot have user-defined SAMPLE_AUGS_NUM greater than the total number of augmentations in user-defined AUGMENTATION_STRING"


## ---------- Set up function to initialize correct model ----------
def set_model(modeltype, args, datamodule):
    modeltype_lower = modeltype.lower()    
    if modeltype_lower == "cnn_s": 
        return CNN_S(datamodule.num_classes, S=args.S, d=args.d)
    elif modeltype_lower == "basset":
        return Basset(datamodule.num_classes, d=args.d)
    elif modeltype_lower == "deepstarr":
        return DeepSTARR(datamodule.num_classes, d=args.d)
    else:
        raise ValueError("unrecognized model type: %s" % modeltype)


## ---------- Set up loss function ----------
losstype_lower = args.loss.lower()
if losstype_lower == "bce": 
    loss = torch.nn.BCELoss()
elif losstype_lower == "mse":
    loss = torch.nn.MSELoss()
else:
    raise ValueError("unrecognized loss function type: %s" % args.loss)



## ---------- Set up function to configure optimizers ----------
def configure_optimizers_sup(model, lr=args.lr, wd=args.wd):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    return {"optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
           }


## ---------- Load data and train model ----------
# Load data and set up DataModule
data_module = H5DataModule(args.data_path, batch_size=args.batch_size)
data_module.setup()

# Set up supervised model, including optimizer
supervised = set_model(args.model_type, args, data_module)
with torch.no_grad():
    temp = torch.zeros(10, data_module.A, data_module.L + args.insert_max) if "insert" in aug_string else data_module.x_train[0:10]
    supervised(temp);

if aug_string == "":
    model_supervised = SupervisedModel(supervised, loss, 
                                       optimizers_configured=configure_optimizers_sup(supervised))
elif args.sample_augs_hard and (args.sample_augs_num == len(augs) or args.sample_augs_num == None): # and aug_string != ""
    model_supervised = SupervisedModelWithAugmentation(supervised, loss,
                                                       optimizers_configured=configure_optimizers_sup(supervised),
                                                       augmentation_string=aug_string,
                                                       invert_min=args.invert_min, invert_max=args.invert_max, 
                                                       delete_min=args.delete_min, delete_max=args.delete_max,
                                                       shift_min=args.shift_min, shift_max=args.shift_max, 
                                                       insert_min=args.insert_min, insert_max=args.insert_max, 
                                                       rc_prob=args.rc_prob, mutate_frac=args.mutate_frac,
                                                       noise_mean=args.noise_mean, noise_std=args.noise_std)
else: # i.e., if aug_string != "" and (args.sample_augs_hard == False or args.sample_augs_num < len(augs))
    model_supervised = SupervisedModelWithStochasticAugmentation(supervised, loss,
                                                                 optimizers_configured=configure_optimizers_sup(supervised),
                                                                 augmentation_string=aug_string,
                                                                 sample_augs_hard=args.sample_augs_hard, 
                                                                 sample_augs_num=args.sample_augs_num,
                                                                 invert_min=args.invert_min, invert_max=args.invert_max, 
                                                                 delete_min=args.delete_min, delete_max=args.delete_max,
                                                                 shift_min=args.shift_min, shift_max=args.shift_max, 
                                                                 insert_min=args.insert_min, insert_max=args.insert_max, 
                                                                 rc_prob=args.rc_prob, mutate_frac=args.mutate_frac,
                                                                 noise_mean=args.noise_mean, noise_std=args.noise_std)

# Fit supervised model
logger = pl.loggers.CSVLogger(output_dir, name=None, version="Log")
callback_topmodel = ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath=output_dir, filename=expt_name+"_Model")
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, auto_select_gpus=True, logger=logger, enable_progress_bar=False, callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience), callback_topmodel])
trainer.fit(model_supervised, datamodule=data_module)


## ---------- Evaluate trained model *at optimal epoch* and record test set performance ----------
checkpoint_topmodel_path = os.path.join(callback_topmodel.dirpath, callback_topmodel.filename + ".ckpt")
model_test = SupervisedModelWithPadding.load_from_checkpoint(checkpoint_path=checkpoint_topmodel_path, model_untrained=supervised, loss_criterion=loss, insert_max=args.insert_max) if "insert" in aug_string else SupervisedModel.load_from_checkpoint(checkpoint_path=checkpoint_topmodel_path, model_untrained=supervised, loss_criterion=loss) # SupervisedModelWithPadding will yield same test_step() results as SupervisedModelWithAugmentation or SupervisedModelWithStochasticAugmentation

dataset_test = torch.utils.data.TensorDataset(data_module.x_test, data_module.y_test)
trainer_test = pl.Trainer(gpus=0)
metrics_test = trainer_test.test(model=model_test, dataloaders=DataLoader(dataset_test, batch_size=data_module.x_test.shape[0]))[0];
if isinstance(loss, torch.nn.modules.loss.BCELoss):
    metrics_formatted = [[metrics_test['test_auroc'], metrics_test['test_aupr']]]
    columns_formatted = ["auroc_supervised", "aupr_supervised"]
elif isinstance(loss, torch.nn.modules.loss.MSELoss):
    metrics_formatted = [[i, metrics_test["test_mse_"+str(i)], metrics_test["test_pearson_r_"+str(i)], metrics_test["test_spearman_rho_"+str(i)]] for i in range(data_module.num_classes)]
    columns_formatted = ["i", "mse_i_supervised", "pearson_r_i_supervised", "spearman_rho_i_supervised"]

performance_df = pd.DataFrame(metrics_formatted, columns=columns_formatted)
performance_file = os.path.join(output_dir, expt_name + "_Performance.tsv")
performance_df.to_csv(performance_file, sep='\t', index=False)


## ---------- Save model configuration/args and trained supervised model as checkpoint ----------
config_output_file = os.path.join(output_dir, expt_name + "_Config.p")
pickle.dump(vars(args), open(config_output_file, "wb" ) )


## ---------- Visualize first-layer filters and save sequence logos ----------
# Select random subset (of size num_select) of x_test for filter visualization
num_select = data_module.x_test.shape[0] if args.filter_viz_subset is None else args.filter_viz_subset
random_selection = torch.randperm(data_module.x_test.shape[0])[:num_select]
x_test_subset = aug_pad_end(data_module.x_test[random_selection], args.insert_max) if "insert" in aug_string else data_module.x_test[random_selection]

# Generate feature maps of first convolutional layer after activation
fmaps = []
def get_output(the_list):
    """get output of layer and put it into list the_list"""
    def hook(model, input, output):
        the_list.append(output.data);
    return hook

model_supervised = model_supervised.eval().to(torch.device("cpu")) # move back to CPU
handle = model_supervised.model.activation1.register_forward_hook(get_output(fmaps))
with torch.no_grad():
    model_supervised.model(x_test_subset);
handle.remove()
fmap = fmaps[0]

# Generate PWMs from feature maps (transposed, to align with TF implementation)
window = math.ceil(model_supervised.model.conv1_filters.shape[-1] / 2.) * 2 # round up to nearest even number
W = activation_pwm( fmap.detach().cpu().numpy().transpose([0,2,1]), x_test_subset.numpy().transpose([0,2,1]), window=window)

# Plot first-layer filters from PWM and save file
fig = plt.figure(figsize=(30.0, 5*(args.d/32)))
fig = plot_filters(W, fig, alphabet=args.alphabet)
outfile = os.path.join(output_dir, expt_name + '_Filters.pdf')
fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

# Save PWMs from first layer filters as pickled object 
pwm_output_file = os.path.join(output_dir, expt_name + "_PWMs.p")
pickle.dump(W, open(pwm_output_file, "wb" ) )

# Generate MEME file
W_clipped = clip_filters(W, threshold=0.5, pad=3)
output_file = os.path.join(output_dir, expt_name + '_Filters.meme')
generate_meme(W_clipped, output_file) 

print("Done!")
