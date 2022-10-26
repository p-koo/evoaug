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
from argparse import Namespace
import pickle
import numpy as np
import pandas as pdtr
from tqdm import tqdm
import matplotlib.pyplot as plt

# EvoAug scripts--check relative paths
from models import *
from zoo import *
from augmentations import aug_pad_end
from datamodules import *
from utils import *


## ---------- Parse arguments and load configuration from original supervised model ----------
parser = argparse.ArgumentParser(description="Fine-tune (without data augmentations) supervised model originally trained with data augmentations on specified dataset", 
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("data_path", metavar="/PATH/TO/DATA/", type=str,
                    help="path to HDF5 file containing data, pre-split into training, validation, and test sets")
parser.add_argument("checkpoint_path", metavar="/PATH/TO/MODEL/", type=str,
                    help="path to trained supervised model weights, in Pytorch checkpoint format (.ckpt)")
parser.add_argument("config_path", metavar="/PATH/TO/CONFIG/", type=str,
                    help="path to configuration for trained supervised model, in pickled file format (.p)")

parser.add_argument("-output_dir", metavar="/PATH/TO/OUTPUT/", type=str, 
                    help="directory where output files (with experiment name appended to results files) will be stored in a new sub-directory named \"Finetune\"; default value of None will result in output directory being set the same one in the trained supervised model configuration (at /PATH/TO/CONFIG/)", 
                    default=None)

parser.add_argument("-lr", metavar="LEARNING_RATE", type=float,
                    help="learning rate for Adam optimizer for fine-tuning", 
                    default=1e-4)
parser.add_argument("-wd", metavar="WEIGHT_DECAY", type=float,
                    help="weight decay (L2 penalty) for Adam optimizer for fine-tuning", 
                    default=1e-6)

parser.add_argument("-batch_size", metavar="BATCH_SIZE", type=int, 
                    help="batch size used in fine-tuning", 
                    default=128)
parser.add_argument("-epochs", metavar="MAX_EPOCHS", type=int, 
                    help="number of epochs to fine-tune model", 
                    default=5)

parser.add_argument("-alphabet", metavar="ALPHABET", type=str, 
                    help="order of nucleotide channels in one-hot encoding of sequences x (e.g., \"ATCG\"), if custom", 
                    default="ACGT")
parser.add_argument("-filter_viz_subset", metavar="SIZE", type=int,
                    help="size of subset of test set to use for convolutional filter visualization; default value of None results in usage of whole test set", 
                    default=None)


# Parse args
args = parser.parse_args()

# Load configuration and calculate relevant variables from original supervised model
config_supervised_dict = pickle.load( open(args.config_path, "rb") )
config_supervised = Namespace(**config_supervised_dict)

expt_name = config_supervised.expt_name
output_dir = os.path.join(args.output_dir, "Finetune") if args.output_dir is not None else os.path.join(config_supervised.output_dir, "Finetune")
make_directory(output_dir) # if necessary, make output directory in which to save results


## ---------- Set up function to initialize correct model ----------
def set_model(modeltype, config, datamodule):
    modeltype_lower = modeltype.lower()    
    if modeltype_lower == "basset":
        return Basset(datamodule.num_classes, d=config.d)
    elif modeltype_lower == "deepstarr":
        return DeepSTARR(datamodule.num_classes, d=config.d)
    else:
        raise ValueError("unrecognized model type: %s" % modeltype)


## ---------- Set up loss function ----------
losstype_lower = config_supervised.loss.lower()
if losstype_lower == "bce": 
    loss = torch.nn.BCELoss()
elif losstype_lower == "mse":
    loss = torch.nn.MSELoss()
else:
    raise ValueError("unrecognized loss function type: %s" % args.loss)


## ---------- Set up function to configure optimizers ----------
def configure_optimizer_ft(model, lr=args.lr, wd=args.wd):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    return {"optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
           }


## ---------- Load data and set up model for fine-tuning ----------
# Load data and set up DataModule
data_module = H5DataModule(args.data_path, batch_size=args.batch_size)
data_module.setup()

# Set up supervised model for fine-tuning, including optimizer
model_untrained = set_model(config_supervised.model_type, config_supervised, data_module)

if "insert" in config_supervised.augs:
    model_finetune = SupervisedModelWithPadding.load_from_checkpoint(checkpoint_path=args.checkpoint_path, 
                                                                     model_untrained=model_untrained, loss_criterion=loss,
                                                                     optimizers_configured=configure_optimizer_ft(model_untrained),
                                                                     insert_max=config_supervised.insert_max)
else: # all other data augmentations do not affect expected input sequence length
    model_finetune = SupervisedModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path, 
                                                          model_untrained=model_untrained, loss_criterion=loss,
                                                          optimizers_configured=configure_optimizer_ft(model_untrained))
    
    
## ---------- Fit supervised fine-tuning model and and save best model ----------
logger = pl.loggers.CSVLogger(output_dir, name=None, version="Log")
callback_topmodel = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
                                    dirpath=output_dir, filename=expt_name+"_Finetune_Model")
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, auto_select_gpus=True, enable_progress_bar=False, 
                     logger=logger, callbacks=[callback_topmodel])
trainer.fit(model_finetune, datamodule=data_module)


## ---------- Evaluate fine-tuned model *at optimal epoch* and record test set performance ----------
checkpoint_topmodel_path = os.path.join(callback_topmodel.dirpath, callback_topmodel.filename + ".ckpt")
model_test = SupervisedModelWithPadding.load_from_checkpoint(checkpoint_path=checkpoint_topmodel_path, model_untrained=model_untrained, loss_criterion=loss, insert_max=config_supervised.insert_max) if "insert" in config_supervised.augs else SupervisedModel.load_from_checkpoint(checkpoint_path=checkpoint_topmodel_path, model_untrained=model_untrained, loss_criterion=loss)

dataset_test = torch.utils.data.TensorDataset(data_module.x_test, data_module.y_test)
trainer_test = pl.Trainer(gpus=0)
metrics_test = trainer_test.test(model=model_test, dataloaders=DataLoader(dataset_test, batch_size=data_module.x_test.shape[0]))[0];

if isinstance(loss, torch.nn.modules.loss.BCELoss):
    metrics_formatted = [[metrics_test['test_auroc'], metrics_test['test_aupr']]]
    columns_formatted = ["auroc_finetune", "aupr_finetune"]
elif isinstance(loss, torch.nn.modules.loss.MSELoss):
    metrics_formatted = [[i, metrics_test["test_mse_"+str(i)], metrics_test["test_pearson_r_"+str(i)], metrics_test["test_spearman_rho_"+str(i)]] for i in range(data_module.num_classes)]
    columns_formatted = ["i", "mse_i_finetune", "pearson_r_i_finetune", "spearman_rho_i_finetune"]

performance_df = pd.DataFrame(metrics_formatted, columns=columns_formatted)
performance_file = os.path.join(output_dir, expt_name + "_Finetune_Performance.tsv")
performance_df.to_csv(performance_file, sep='\t', index=False)


## ----- Visualize first-layer filters and save sequence logos -----
# Select random subset (of size num_select) of x_test for filter visualization
num_select = data_module.x_test.shape[0] if config_supervised.filter_viz_subset is None else config_supervised.filter_viz_subset
random_selection = torch.randperm(data_module.x_test.shape[0])[:num_select]
x_test_subset = aug_pad_end(data_module.x_test[random_selection], config_supervised.insert_max) if "insert" in config_supervised.augs else data_module.x_test[random_selection]

# Generate feature maps of first convolutional layer after activation
fmaps = []
def get_output(the_list):
    """get output of layer and put it into list the_list"""
    def hook(model, input, output):
        the_list.append(output.data);
    return hook

model_finetune = model_finetune.eval().to(torch.device("cpu")) # move back to CPU
handle = model_finetune.model.activation1.register_forward_hook(get_output(fmaps))
with torch.no_grad():
    model_finetune.model(x_test_subset);
handle.remove()
fmap = fmaps[0]

# Generate PWMs from feature maps (transposed, to align with TF implementation)
window = math.ceil(model_finetune.model.conv1_filters.shape[-1] / 2.) * 2 # round up to nearest even number
W = activation_pwm( fmap.detach().cpu().numpy().transpose([0,2,1]), x_test_subset.numpy().transpose([0,2,1]), window=window)

# Plot first-layer filters from PWM and save file
fig = plt.figure(figsize=(30.0, 5*(config_supervised.d/32)))
fig = plot_filters(W, fig, alphabet=args.alphabet)
outfile = os.path.join(output_dir, expt_name + '_Finetune_Filters.pdf')
fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

# Save PWMs from first layer filters as pickled object 
pwm_output_file = os.path.join(output_dir, expt_name + "_Finetune_PWMs.p")
pickle.dump(W, open(pwm_output_file, "wb" ) )

# Generate MEME file
W_clipped = clip_filters(W, threshold=0.5, pad=3)
output_file = os.path.join(output_dir, expt_name + '_Finetune_Filters.meme')
generate_meme(W_clipped, output_file) 


print("Done!")
