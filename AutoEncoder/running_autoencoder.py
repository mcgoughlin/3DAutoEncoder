import os
os.environ['OV_DATA_BASE'] = '/media/mcgoug01/nvme/AE_practise/'

from ovseg.model.AutoEncoderModel import AutoEncoderModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.AutoEncoderPreprocessing import AutoEncoderPreprocessing
os.environ['OV_DATA_BASE'] = '/media/mcgoug01/nvme/AE_practise/'

import gc
import torch
import sys


data_name = 'kits_ncct'
spacing = 4.0

preprocessed_name = '{}mm_AE'.format(spacing)
model_name = 'test'

vfs = [0]
lb_classes = [1]

target_spacing=[spacing,spacing,spacing]
kits19ncct_dataset_properties ={
    		'median_shape' : [104., 512., 512.],
    		'median_spacing' : [2.75,      0.8105, 0.8105],
    		'fg_percentiles' : [-118, 136 ],
    		'percentiles' : [0.5, 99.5],
    		'scaling_foreground' : [ 42.94977,  11.57459],
    		'n_fg_classes' : 1,
    		'scaling_global' : [ 510.60403, -431.1344],
    		'scaling_window' : [ 68.93295,  -65.79061]}




# prep = AutoEncoderPreprocessing(apply_resizing=True, 
#                                     apply_pooling=False, 
#                                     apply_windowing=True,
#                                     lb_classes=lb_classes,
#                                     target_spacing=target_spacing,
#                                     scaling = [ 42.94977,  11.57459],
#                                     window = [-118, 136 ],
#                                     dataset_properties = kits19ncct_dataset_properties)

# prep.initialise_preprocessing()

# prep.preprocess_raw_data(raw_data=data_name,
#                           preprocessed_name=preprocessed_name,
#                           dist_flag=False)
#patch dimension must be divisible by respective (((kernel_dimension+1)//2)^depth)/2
#Patch size dictates input size to CNN: input dim (metres) = patch_size*target_spacing/1000
#finally, depth and conv kernel size dictate attentive area - importantly different to input size:
# attentive_area (in each dimension, metres) = input size / bottom encoder spatial dim
#                                           = ((((kernel_dimension+1)//2)^depth)/2)*target_spacing/1000
z_to_xy_ratio = 1
larger_res_encoder = True
n_fg_classes = 1
patch_size = [32,32,32]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=False)


model_params['network']['kernel_sizes'] =5*[(3,3,3)]
model_params['network']['norm'] = 'inst'
model_params['network']['in_channels']=1
model_params['network']['filters']=8
model_params['network']['filters_max']=32
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

lr=0.0001
# dist_flag settings
model_params['data']['folders'] = ['images']#, 'masks']
model_params['data']['keys'] = ['image']#, 'mask']

model_params['training']['num_epochs'] = 50
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': lr,
                                            'betas': (0.95, 0.9),
                                            'eps': 1e-08}
model_params['training']['lr_params'] = {'n_warmup_epochs': 5, 'lr_max': 0.005}
model_params['data']['trn_dl_params']['epoch_len']=500
model_params['data']['trn_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['data']['val_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['training']['lr_schedule'] = 'lin_ascent_log_decay'
model_params['training']['lr_exponent'] = 4
model_params['training']['loss_params'] = {'loss_names':['mse_loss'], 
                                           'loss_weights':[1]}
model_params['data']['trn_dl_params']['batch_size']=16
model_params['data']['val_dl_params']['epoch_len']=50
# model_params['postprocessing'] = {'mask_with_reg': True}

for vf in vfs:
    model = AutoEncoderModel(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=preprocessed_name,
                                model_name=model_name,
                                model_parameters=model_params)
    torch.cuda.empty_cache()
    gc.collect()

    model.training.train()
    torch.cuda.empty_cache()
    gc.collect()
    
    model.eval_validation_set()
    torch.cuda.empty_cache()
    gc.collect()
