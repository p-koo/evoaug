User Guide
==========

.. _installation:

Installation
------------

To use EvoAug, first install it using pip:

.. code-block:: console

   pip install evoaug

Example
-------

Import evoaug:

.. code-block:: python

   from evoaug import evoaug, augment
   import lightning.pytorch as pl


Define PyTorch model and modeling choices:

.. code-block:: python

   model = "DEFINE PYTORCH MODEL"
   loss = "DEFINE PYTORCH LOSS"
   optimizer_dict = "DEFINE OPTIMIZER OR OPTIMIZER DICT"
   ckpt_aug_path = "path-to-aug-checkpoint.ckpt"
   ckpt_finetune_path = "path-to-finetune-checkpoint.ckpt"


Train model with augmentations:

.. code-block:: python

   augment_list = [
      augment.RandomDeletion(delete_min=0, delete_max=20),
      augment.RandomRC(rc_prob=0.5),
      augment.RandomInsertion(insert_min=0, insert_max=20),
      augment.RandomTranslocation(shift_min=0, shift_max=20),
      augment.RandomMutation(mut_frac=0.05),
      augment.RandomNoise(noise_mean=0, noise_std=0.2),
   ]

   robust_model = evoaug.RobustModel(
      model,
      criterion=loss,
      optimizer=optimizer_dict,
      augment_list=augment_list,
      max_augs_per_seq=2,  # maximum number of augmentations per sequence
      hard_aug=True,  # use max_augs_per_seq, otherwise sample randomly up to max
      inference_aug=False,  # if true, keep augmentations on during inference time
   )

   # set up callback
   callback_topmodel = pl.callbacks.ModelCheckpoint(
      monitor="val_loss", save_top_k=1, dirpath=output_dir, filename=ckpt_aug_path
   )

   # train model
   trainer = pl.Trainer(
      accelerator="gpu",
      devices=1,
      max_epochs=100,
      logger=None,
      callbacks=["ADD CALLBACKS", "callback_topmodel"],
   )

   # pre-train model with augmentations
   trainer.fit(robust_model, datamodule=data_module)

   # load best model
   robust_model = evoaug.load_model_from_checkpoint(robust_model, ckpt_aug_path)


Fine-tune model without augmentations:

.. code-block:: python

   # set up fine-tuning
   robust_model.finetune = True
   robust_model.optimizer = "set up optimizer for fine-tuning"

   # set up callback
   callback_topmodel = pl.callbacks.ModelCheckpoint(
      monitor="val_loss",
      save_top_k=1,
      dirpath=output_dir,
      filename=ckpt_finetune_path,
   )

   # set up pytorch lightning trainer
   trainer = pl.Trainer(
      accelerator="gpu",
      devices=1,
      max_epochs=100,
      logger=None,
      callbacks=["ADD CALLBACKS", "callback_topmodel"],
   )

   # fine-tune model
   trainer.fit(robust_model, datamodule=data_module)

   # load best fine-tuned model
   robust_model = evoaug.load_model_from_checkpoint(robust_model, ckpt_finetune_path)



Examples on Google Colab
------------------------

DeepSTARR analysis:

.. code-block:: python

   https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf?usp=sharing


ChIP-seq analysis:

.. code-block:: python

   https://colab.research.google.com/drive/1GZ8v4Tq3LQMZI30qvdhF7ZW6Kf5GDyKX?usp=sharing




