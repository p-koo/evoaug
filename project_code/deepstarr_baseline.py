import os
import numpy as np
import torch
import pytorch_lightning as pl
from six.moves import cPickle
from evoaug import robust_model, moana, utils
from model_zoo import DeepSTARR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load data
expt_name = 'DeepSTARR'
data_path = '../data'
filepath = os.path.join(data_path, expt_name + '_data.h5')
data_module = utils.H5DataModule(filepath, batch_size=100, lower_case=False)


output_dir = '../results/deepstarr'
utils.make_directory(output_dir)

num_trials = 5 


trial_results = []
for trial in range(num_trials):

    deepstarr = DeepSTARR(data_module.y_train.shape[-1]).to(device)
    loss = torch.nn.MSELoss()
    optimizer_dict = utils.configure_optimizer(deepstarr, 
                                               lr=0.001, 
                                               weight_decay=1e-6, 
                                               decay_factor=0.1, 
                                               patience=5, 
                                               monitor='val_loss')

    robust_deepstarr = robust_model.RobustModel(deepstarr,
                                   criterion=loss,
                                   optimizer=optimizer_dict, 
                                   augment_list=[])

    # create pytorch lightning trainer
    ckpt_aug_path = expt_name+"_baseline_"+str(trial)
    callback_topmodel = pl.callbacks.ModelCheckpoint(monitor='val_loss', 
                                                     save_top_k=1, 
                                                     dirpath=output_dir, 
                                                     filename=ckpt_aug_path)
    callback_es = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=10)
    trainer = pl.Trainer(gpus=1, max_epochs=100, auto_select_gpus=True, logger=None, 
                        callbacks=[callback_es, callback_topmodel])

    # fit model
    trainer.fit(robust_deepstarr, datamodule=data_module)

    # load checkpoint for model with best validation performance
    robust_deepstarr = robust_model.load_model_from_checkpoint(robust_deepstarr, ckpt_aug_path+'.ckpt')

    # evaluate best model
    pred = utils.get_predictions(robust_deepstarr, data_module.x_test, batch_size=100)
    results = utils.evaluate_model(data_module.y_test, pred, task='regression')   # task is 'binary' or 'regression'

    # store results
    trial_results.append(results)

# save results
with open(os.path.join(output_dir, 'deepstarr_baseline.pickle'), 'wb') as fout:
    cPickle.dump(trial_results, fout)


    