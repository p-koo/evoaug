import os
import numpy as np
import torch
import pytorch_lightning as pl
from six.moves import cPickle
from evoaug import robust_model, moana, utils
from model_zoo import CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

output_dir = '../results/chipseq'
utils.make_directory(output_dir)


expt_names = ['GABPA', 'MAX', 'BACH1', 'REST', 'SRF', 'ZNF24', 'ELK1']
num_trials = 5 

for expt_name in expt_names:

    data_path = '../data'
    filepath = os.path.join(data_path, expt_name + '_200.h5')
    data_module = utils.H5DataModule(filepath, batch_size=100, lower_case=True)

    trial_aug_results = []
    trial_finetune_results = []
    for trial in range(num_trials):

        cnn = CNN(data_module.y_train.shape[-1]).to(device)
        loss = torch.nn.BCELoss()
        optimizer_dict = utils.configure_optimizer(cnn, 
                                                   lr=0.001, 
                                                   weight_decay=1e-6, 
                                                   decay_factor=0.1, 
                                                   patience=5, 
                                                   monitor='val_loss')
        augment_list = [
            augment.RandomRC(rc_prob=0.5),
            augment.RandomDeletion(delete_min=0, delete_max=20),
            augment.RandomInsertion(insert_min=0, insert_max=20),
            augment.RandomTranslocation(shift_min=0, shift_max=20),
            augment.RandomNoise(noise_mean=0, noise_std=0.2),
            augment.RandomMutation(mutate_frac=0.05),
        ]

        robust_cnn = robust_model.RobustModel(cnn,
                                       criterion=loss,
                                       optimizer=optimizer_dict, 
                                       augment_list=augment_list,
                                       max_augs_per_seq=3, 
                                       hard_aug=False)

        # create pytorch lightning trainer
        ckpt_aug_path = expt_name+"_aug_"+str(trial)
        callback_topmodel = pl.callbacks.ModelCheckpoint(monitor='val_loss', 
                                                         save_top_k=1, 
                                                         dirpath=output_dir, 
                                                         filename=ckpt_aug_path)
        callback_es = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=10)
        trainer = pl.Trainer(gpus=1, max_epochs=100, auto_select_gpus=True, logger=None, 
                            callbacks=[callback_es, callback_topmodel])

        # fit model
        trainer.fit(robust_cnn, datamodule=data_module)

        # load checkpoint for model with best validation performance
        robust_cnn = robust_model.load_model_from_checkpoint(robust_cnn, ckpt_aug_path+'.ckpt')

        # evaluate best model
        pred = utils.get_predictions(robust_cnn, data_module.x_test, batch_size=100)
        aug_results = utils.evaluate_model(data_module.y_test, pred, task='binary')   


        ## ---------- Fine tune analysis ----------

        # Load best EvoAug model from checkpoint
        robust_cnn.finetune = True
        robust_cnn.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, robust_cnn.model.parameters()),
                                                   lr=0.0001, weight_decay=1e-6)

        # set up trainer for fine-tuning
        ckpt_finetune_path = expt_name+"_finetune_"+str(trial)
        callback_topmodel = pl.callbacks.ModelCheckpoint(monitor='val_loss', 
                                                         save_top_k=1, 
                                                         dirpath=output_dir, 
                                                         filename=ckpt_finetune_path)
        trainer = pl.Trainer(gpus=1, max_epochs=5, auto_select_gpus=True, logger=None, 
                            callbacks=[callback_topmodel])

        # Fine-tune model
        trainer.fit(robust_cnn, datamodule=data_module)

        # load checkpoint for model with best validation performance
        robust_cnn = robust_model.load_model_from_checkpoint(robust_cnn, ckpt_finetune_path+'.ckpt')

        # evaluate best model
        pred = utils.get_predictions(robust_cnn, data_module.x_test, batch_size=100)
        finetune_results = utils.evaluate_model(data_module.y_test, pred, task='binary') 

        # store results
        trial_aug_results.append(aug_results)
        trial_finetune_results.append(finetune_results)

    # save results
    with open(os.path.join(output_dir, expt_name+'_analysis.pickle'), 'wb') as fout:
        cPickle.dump(all_finetune_results, fout)
        cPickle.dump(all_aug_results, fout)



    