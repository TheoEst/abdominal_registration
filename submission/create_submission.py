#!/usr/bin/python
import glob
import numpy as np
import os
from scipy.ndimage.interpolation import zoom as zoom
import shutil
import sys
import pandas as pd
    
def main():
    INPUT_FOLDER = sys.argv[1]
    print(INPUT_FOLDER)
    OUTPUT_FOLDER = INPUT_FOLDER + '_compressed'

    def create_empty_grid(task, shape):
    
        dataframe = pd.read_csv('./datasets/task_0{}_pairs_val.csv'.format(task))
         
        for i in range(len(dataframe)):
            
            fixed = dataframe.loc[i]['fixed']
            moving = dataframe.loc[i][' moving']
            
            disp = np.zeros(shape, dtype='float16')
            path = OUTPUT_FOLDER + '/task_0{}/disp_{:04d}_{:04d}.npz'.format(task,
                                                                            int(fixed),
                                                                            int(moving))
            np.savez_compressed(path, disp)
        
    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_01'))
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_02'))
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_03'))
        os.mkdir(os.path.join(OUTPUT_FOLDER, 'task_04'))
    
    # TASK 1
    files = glob.glob(os.path.join(INPUT_FOLDER, 'task_01', '*.npy'))
    if len(files) == 0:
        shape = (3, 256 //2, 256//2, 288//2)
        create_empty_grid(1, shape)
        
    else:
        for file in files:
            print('compressing {}...'.format(file))
            disp = np.load(file) #expects shape 3x256x256x288
            disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
            disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
            disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
            disp = np.array((disp_x, disp_y, disp_z))
            np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
       
    # TASK 2
    files = glob.glob(os.path.join(INPUT_FOLDER, 'task_02', '*.npy'))
    if len(files) == 0:
        shape = (3, 192 //2, 192//2, 208//2)
        create_empty_grid(2, shape)
        
    else:
        for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_02', '*.npy')):
            print('compressing {}...'.format(file))
            disp = np.load(file) #expects shape 3x192x192x208
            disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
            disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
            disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
            disp = np.array((disp_x, disp_y, disp_z))
            np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
    
    # TASK 3
    files = glob.glob(os.path.join(INPUT_FOLDER, 'task_03', '*.npy'))
    print(files)
    if len(files) == 0:
        shape = (3, 192 //2, 160//2, 256//2)
        create_empty_grid(3, shape)
    else:
        for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_03', '*.npy')):
            print('compressing {}...'.format(file))
            disp = np.load(file) #expects shape 3x192x160x256
            print(disp.shape)
            disp_x = zoom(disp[0], 0.5, order=2).astype('float16')
            disp_y = zoom(disp[1], 0.5, order=2).astype('float16')
            disp_z = zoom(disp[2], 0.5, order=2).astype('float16')
            disp = np.array((disp_x, disp_y, disp_z))
            np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
    
    # TASK 4
    files = glob.glob(os.path.join(INPUT_FOLDER, 'task_04', '*.npy'))
    if len(files) == 0:
        shape = (3, 64, 64, 64)
        create_empty_grid(4, shape)
    else:
        for file in glob.glob(os.path.join(INPUT_FOLDER, 'task_04', '*.npy')):
            print('compressing {}...'.format(file))
            disp = np.load(file) #expects shape 3x64x64x64
            disp = disp.astype('float16')
            np.savez_compressed(file.replace(INPUT_FOLDER, OUTPUT_FOLDER).replace('npy', 'npz'), disp)
        
    shutil.make_archive(INPUT_FOLDER + '_submission', 'zip', OUTPUT_FOLDER)
    
    print('...finished creating submission.')

if __name__ == "__main__":
    main()
