# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:14:09 2020

@author: t_estienne
"""
from evalutils.io import CSVLoader, ImageLoader
import nibabel as nib
import numpy as np
import scipy.ndimage
from scipy.ndimage.interpolation import map_coordinates, zoom
from surface_distance import compute_dice_coefficient, compute_robust_hausdorff, compute_surface_distances
from pandas import DataFrame, MultiIndex
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet


class NiftiLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return nib.load(str(fname))

    @staticmethod
    def hash_image(image):
        return hash(image.get_fdata().tostring())
    
class NumpyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return np.load(str(fname))['arr_0']

    @staticmethod
    def hash_image(image):
        return hash(image.tostring())

def main():
    
    data_path = './data/'
    submission_path = './save/submission/'
    save_path = './save/evaluation/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
 
    DEFAULT_GROUND_TRUTH_PATH = data_path + 'L2R_Task3_AbdominalCT/Training/label/'
    submission_name = sys.argv[1] 
    DEFAULT_INPUT_PATH = submission_path + submission_name + '/task_03/'
    pairs_path = './datasets/task_03_pairs_val.csv'
    DEFAULT_EVALUATION_OUTPUT_FILE_PATH = save_path + submission_name + '.csv'
    
    EvalVal(DEFAULT_INPUT_PATH, 
            DEFAULT_EVALUATION_OUTPUT_FILE_PATH,
            pairs_path, DEFAULT_GROUND_TRUTH_PATH).evaluate()
    
class EvalVal():
    def __init__(self, DEFAULT_INPUT_PATH, 
                 DEFAULT_EVALUATION_OUTPUT_FILE_PATH,
                 pairs_path, DEFAULT_GROUND_TRUTH_PATH
                 ):
        
        self.DEFAULT_INPUT_PATH = DEFAULT_INPUT_PATH
        self.output_file = DEFAULT_EVALUATION_OUTPUT_FILE_PATH
        self.pairs_path = pairs_path
        self.DEFAULT_GROUND_TRUTH_PATH = DEFAULT_GROUND_TRUTH_PATH
        
        self.csv_loader = CSVLoader()
        self.nifti_loader = NiftiLoader()
        self.numpy_loader = NumpyLoader()
        
        self.pairs_task_03 = DataFrame()
        self.segs_task_03 = DataFrame()
        self.disp_fields_task_03 = DataFrame()
        self.cases_task_03 = DataFrame()
        
    
    def evaluate(self):

        self.load_task_03()
        self.merge_ground_truth_and_predictions_task_03()
        self.score_task_03()
              
        self.save()
           
    def load_task_03(self):
        self.pairs_task_03 = self.load_pairs(self.pairs_path)
        self.segs_task_03 = self.load_segs_task_03()
        self.disp_fields_task_03 = self.load_disp_fields(self.pairs_task_03,
                                                         self.DEFAULT_INPUT_PATH, 
                                                         np.array([3, 96, 80, 128]))
        

    def load_segs_task_03(self):
        cases = None
        
        indices = []
        for _, row in self.pairs_task_03.iterrows():
            indices.append(row['fixed'])
            indices.append(row['moving'])
        indices = np.array(indices)

        for i in np.unique(indices):
            case =  self.nifti_loader.load(fname=self.DEFAULT_GROUND_TRUTH_PATH +  'label{:04d}.nii.gz'.format(i))
        
            if cases is None:
                cases = case
                index = [i]
            else:
                cases += case
                index += [i]

        return DataFrame(cases, index=index)
    
  
    def merge_ground_truth_and_predictions_task_03(self):
        cases = []
        for _, row in self.pairs_task_03.iterrows():
            case = {'seg_fixed' : self.segs_task_03.loc[row['fixed']],
                    'seg_moving' : self.segs_task_03.loc[row['moving']],
                    'disp_field' : self.disp_fields_task_03.loc[(row['fixed'], row['moving'])]}
            cases += [case]
        self.cases_task_03 = DataFrame(cases)
        
    
    def score_task_03(self):
        self.cases_results_task_03 = DataFrame()
        for idx, case in tqdm(self.cases_task_03.iterrows()):
            self.cases_results_task_03 = self.cases_results_task_03.append(self.score_case_task_03(idx=idx, case=case), ignore_index=True)
        self.aggregate_results_task_03 = self.score_aggregates_task_03()
        
        
    def plot(self, moving, fixed, moving_warped):
        
        fig, ax = plt.subplots(3, 3, gridspec_kw={'wspace': 0, 'hspace': 0.02,
                                                  'top': 0.93, 'bottom': 0.01,
                                                  'left': 0.01, 'right': 0.99})
        x, y, z = 96, 80, 128
        mask_kwargs = {'vmin':0, 'vmax':14}
            
        ax[0, 0].imshow(moving[x, :, :], **mask_kwargs)
        ax[1, 0].imshow(moving[:, y, :], **mask_kwargs)
        ax[2, 0].imshow(moving[:, :, z], **mask_kwargs)
        
        ax[0, 1].imshow(fixed[x, :, :], **mask_kwargs)
        ax[1, 1].imshow(fixed[:, y, :], **mask_kwargs)
        ax[2, 1].imshow(fixed[:, :, z], **mask_kwargs)
        
        ax[0, 2].imshow(moving_warped[x, :, :], **mask_kwargs)
        ax[1, 2].imshow(moving_warped[:, y, :], **mask_kwargs)
        ax[2, 2].imshow(moving_warped[:, :, z], **mask_kwargs)
        
        titles = ['Moving', 'Fixed', 'Warped']
        for j in range(3):
            ax[0, j].set_title(titles[j])
        
        fig.savefig('random')
            
    def score_case_task_03(self, *, idx, case):
        fixed_path = case['seg_fixed']['path']
        moving_path = case['seg_moving']['path']
        disp_field_path = case['disp_field']['path']

        fixed =  self.nifti_loader.load_image(fixed_path).get_fdata()
        moving =  self.nifti_loader.load_image(moving_path).get_fdata()
        disp_field = self.numpy_loader.load_image(disp_field_path).astype('float32')

        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
       
        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 15).clip(0, 1000000000)
        log_jac_det = np.log(jac_det)
        
        D, H, W = fixed.shape

        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        moving_warped = map_coordinates(moving, identity + disp_field, order=0)
        
        #self.plot(moving, fixed, moving_warped)
        
        # dice
        dice = 0
        count = 0
        for i in range(1, 14):
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                continue
            organ_dice = compute_dice_coefficient((fixed==i), (moving_warped==i))
            dice += organ_dice
            #print("{} :: {}".format(i, organ_dice))
            count += 1
        dice /= count
        
        # hd95
        hd95 = 0
        count = 0
        for i in range(1, 14):
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                continue
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.)
            count += 1
        hd95 /= count
        
        score = {'DiceCoefficient' : dice,
                'HausdorffDistance95' : hd95,
                'LogJacDetStd' : log_jac_det.std()}
        print(score)
        return score
    

    def score_aggregates_task_03(self):
        aggregate_results = {}

        for col in self.cases_results_task_03.columns:
            aggregate_results[col] = self.aggregate_series_task_03(series=self.cases_results_task_03[col])

        return aggregate_results
    
    
    def aggregate_series_task_03(self, *, series):
        series_summary = {}
        
        series_summary['mean'] = series.mean()
        series_summary['std'] = series.std()
        series_summary['30'] = series.quantile(.3)
        
        return series_summary
    
    def load_pairs(self, fname):
        return DataFrame(self.csv_loader.load(fname=fname))
    
    def load_disp_fields(self, pairs, folder, expected_shape):
        cases = None
        
        for _, row in pairs.iterrows():
            fname = folder + 'disp_{:04d}_{:04d}.npz'.format(row['fixed'], row['moving'])
            
            if os.path.isfile(fname):
                case = self.numpy_loader.load(fname=fname)
                
                disp_field = self.numpy_loader.load_image(fname=fname)
                dtype = disp_field.dtype
                if not dtype == 'float16':
                    print('DTYPE Error {} ::: {}'.format(fname, dtype))
                    
                shape = np.array(disp_field.shape)
                if not (shape==expected_shape).all():
                    print('SHAPE Error {} ::: {} / {}'.format(fname, shape, expected_shape))

                if cases is None:
                    cases = case
                    index = [(row['fixed'], row['moving'])]
                else:
                    cases += case
                    index.append((row['fixed'], row['moving']))
            else:
                print('MISSING FILES Error {}'.format(fname))
                
        return DataFrame(cases, index=MultiIndex.from_tuples(index))
    
    def metrics(self):
        return {
                "task_03" : {
                    "case": self.cases_results_task_03.to_dict(),
                    "aggregates": self.aggregate_results_task_03,
                            }
               }
    
    def save(self):
        with open(self.output_file, "w") as f:
            f.write(json.dumps(self.metrics()))
            
            
##### main #####
if __name__ == "__main__":
    main()
