import numpy as np
import torch
from ovseg.utils.torch_np_utils import check_type, maybe_add_channel_dim
from skimage.measure import label
from skimage.transform import resize
from torch.nn.functional import interpolate
from scipy.ndimage.morphology import binary_fill_holes
from ovseg.utils.torch_morph import morph_cleaning


class AutoEncoderPostprocessing(object):

    def __init__(self,
                 apply_small_component_removing=False,
                 volume_thresholds=None,
                 remove_2d_comps=True,
                 remove_comps_by_volume=False,
                 mask_with_reg=False,
                 lb_classes=None,
                 use_fill_holes_2d=False,
                 use_fill_holes_3d=False,
                 keep_only_largest=False,
                 apply_morph_cleaning=False):

        pass
        
    def postprocess_volume(self, volume, reg=None, spacing=None, orig_shape=None):
        '''
        postprocess_volume(volume, orig_shape=None)

        Applies the following for post processing:
            - resizing to original voxel spacing (if given)
            - applying argmax to go from hard to soft labels
            - removing small connected components (if set to true)

        Parameters
        ----------
        volume : array tensor
            volume with soft segmentation/ output of the CNN
        orig_shape : len 3, optional
            if out_shape is given the volume is resized to original shape
            before any other postprocessing is done

        Returns
        -------
        postprocessed hard segmentation labels

        '''

        # first let's check if the input is right
        is_np, _ = check_type(volume)
        inpt_shape = np.array(volume.shape)
        if len(inpt_shape) != 4:
            raise ValueError('Expected 4d volume of shape '
                             '[n_channels, nx, ny, nz].')
        
        
        # first fun step: let's reshape to original size
        # before going to hard labels
        if orig_shape is not None:
            if np.any(orig_shape != inpt_shape):
                orig_shape = np.array(orig_shape)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        if is_np:
                            volume = torch.from_numpy(volume).to('cuda').type(torch.float)
                        size = [int(s) for s in orig_shape]
                        volume = interpolate(volume.unsqueeze(0),
                                             size=size,
                                             mode='trilinear')[0]

                else:
                    if not is_np:
                        volume = volume.cpu().numpy()
                    volume = np.stack([resize(volume[c], orig_shape, 1)
                                       for c in range(volume.shape[0])])

        
        # now change to CPU and numpy
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
            
                            
        # print(volume.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(volume[0,0])
        
        return volume

    def postprocess_data_tpl(self, data_tpl, prediction_key, reg=None):

        pred = data_tpl[prediction_key]

        spacing = data_tpl['spacing'] if 'spacing' in data_tpl else None
        if 'orig_shape' in data_tpl:
            # the data_tpl has preprocessed data.
            # predictions in both preprocessed and original shape will be added
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               reg,
                                                               spacing=spacing,
                                                               orig_shape=None)
            spacing = data_tpl['orig_spacing'] if 'orig_spacing' in data_tpl else None
            shape = data_tpl['orig_shape']
            data_tpl[prediction_key+'_orig_shape'] = self.postprocess_volume(pred,
                                                                             reg,
                                                                             spacing=spacing,
                                                                             orig_shape=shape)
        else:
            # in this case the data is not preprocessed
            orig_shape = data_tpl['label'].shape
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               reg,
                                                               spacing=spacing,
                                                               orig_shape=orig_shape)
        return data_tpl

 