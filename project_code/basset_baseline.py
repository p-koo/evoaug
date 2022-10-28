import os
import numpy as np
import torch
import pytorch_lightning as pl
from six.moves import cPickle
import evoaug
from evoaug import utils, augment, robust_model
from model_zoo import Basset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load data
expt_name = 'Basset'
data_path = '../data'
filepath = os.path.join(data_path, expt_name + '_data.h5')
data_module = evoaug.utils.H5DataModule(filepath, batch_size=100, lower_case=False)


output_dir = '../results/basset'
utils.make_directory(output_dir)

num_trials = 5 


trial_results = []
for trial in range(num_trials):

    basset = Basset(data_module.y_train.shape[-1]).to(device)
    loss = torch.nn.BCELoss()
    optimizer_dict = utils.configure_optimizer(basset, 
                                               lr=0.001, 
                                               weight_decay=1e-6, 
                                               decay_factor=0.1, 
                                               patience=5, 
                                               monitor='val_loss')

    augment_list = [
    ]
    robust_basset = robust_model.RobustModel(basset,
                                   criterion=loss,
                                   optimizer=optimizer_dict, 
                                   augment_list=augment_list)

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
    trainer.fit(robust_basset, datamodule=data_module)

    # load checkpoint for model with best validation performance
    robust_basset = robust_model.load_model_from_checkpoint(robust_basset, ckpt_aug_path+'.ckpt')

    # evaluate best model
    pred = utils.get_predictions(robust_basset, data_module.x_test, batch_size=100)
    results = utils.evaluate_model(data_module.y_test, pred, task='binary')   

    # store results
    trial_results.append(results)


    ## ---------- Visualize first-layer filters and save sequence logos ----------

    # Get feature maps of first convolutional layer after activation
    x_test_subset = data_module.x_test[:10000]
    fmaps = utils.get_fmaps(robust_basset, x_test_subset)

    # Generate PWMs from feature maps (transposed, to align with TF implementation)
    ppm = moana.activation_pwm( fmaps, x_test_subset.numpy().transpose([0,2,1]), window=21)  

    # Save PPMs from first layer filters as pickled object 
    ppm_output_file = os.path.join(output_dir, expt_name+'_baseline_ppm_'+str(trial)+'.pickle')
    cPickle.dump(ppm, open(ppm_output_file, "wb" ) )

    # Generate MEME file
    ppm_clipped = moana.clip_filters(ppm, threshold=0.5, pad=3)
    output_file = os.path.join(output_dir, expt_name+'_baseline_filters_'+str(trial)+'.meme')
    moana.generate_meme(ppm_clipped, output_file) 

# save results
with open(os.path.join(output_dir, expt_name+'_baseline.pickle'), 'wb') as fout:
    cPickle.dump(trial_results, fout)


    


