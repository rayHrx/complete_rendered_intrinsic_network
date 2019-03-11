import os, torch, torch.utils.data, scipy.misc, numpy as np, pdb
from .utils import *

'''
directory : base path of datasets 
    e.g., 'outputs/'

datasets : comma-separated string of dataset names within directory 
    e.g., 'motorbike,airplane,bottle'

selections : list of intrinsic images to load 
    for decomposer, ['input', 'mask', 'albedo', 'depth', 'normals', 'lights']
    for shader, ['normals', 'lights', 'shading']

size_per_dataset : number of images to load from each dataset

array : name of lighting parameter array
'''
#train_set = pipeline.IntrinsicDataset(args.data_path, args.inference_set, args.inference, array=args.array, size_per_dataset=args.num_train)
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, directory, datasets, selections,size_per_dataset=1,rel_path=''):
        self.directory = directory
        self.data_files = []
        ## get list of relative paths to all dataset folders
        self.datasets = [os.path.join(directory, dataset) for dataset in datasets.split(',')]
        self.selections = selections
        self.rel_path = rel_path
        self.size_per_dataset = size_per_dataset

        for dataset in self.datasets:
            ## list of images for this object category
            self.set_specific = []
            for sel in self.selections:             
                files = self.__find_sort_files(dataset, sel)[:size_per_dataset] 
                self.set_specific.append(files)
            self.data_files.append(self.set_specific)
        
        ## merge the images from all object categories
        self.data_files = [self.__merge(lists) for lists in zip(*self.data_files)]

    ## sel : str
    ## find images in dataset folder with sel in name.
    ## image names will be of the form <ind>_sel.png
    ## sorts image names and returns the sorted list of relative filepaths.
    def __find_sort_files(self, dataset, sel):
        files = [fname for fname in os.listdir(self.rel_path + dataset) if sel + '.png' in fname and 'sphere' not in fname]
        files = sorted(files, key=lambda fname: int(fname.split('_')[0]) ) 
        files = [os.path.join(self.rel_path + dataset, fname) for fname in files]  
        return files

    ## merge selection from different datasets,
    ## e.g., merge all reflectance images from 
    ## motorbikes, airplanes, and bottles

    ## inputs are represented as dicts with reflectance and shading filenames
    ## lights are represented as np arrays
    ## other intrinsic images as list of filenames 
    def __merge(self, lists):
        if type(lists[0]) == np.ndarray:
            return np.concatenate(lists, 0)

        elif type(lists[0]) == dict:
            merged = {}
            for key in lists[0].keys():
                merged[key] = [i for subdict in lists for i in subdict[key]]
            return merged
        else:
            return [i for sublist in lists for i in sublist]

    ## read image as C x M x N array in range [0, 1]
    def __read_image(self, path):
        img = scipy.misc.imread(path)
        if img.shape[-1] == 4:
            img = img[:,:,:-1]
        img = img.transpose(2,0,1) / 255.
        return img

    def __getitem__(self, idx):
        outputs = []
        for ind, sel in enumerate(self.selections):

            ## lights : simply index into array
            if sel == 'lights':
                out = self.data_files[ind][idx]
                # out = self.data_files[ind][idx,:]

            ## composite : pointwise multiplication between reflectance and shading
            elif sel == 'input':
                path = self.data_files[ind][idx]
                out = self.__read_image(path)
                # reflectance_path = self.data_files[ind]['reflectance'][idx]
                # shading_path = self.data_files[ind]['shading'][idx]
                # reflectance = self.__read_image(reflectance_path)
                # shading = self.__read_image(shading_path)
                # out = reflectance * shading
            
            ## other intrinsic images : read image from disk
            else:
                path = self.data_files[ind][idx]
                out = self.__read_image(path)
                ## shading is 1 channel
                if sel == 'shading':
                    out = out[0,:,:]
                ## depth is 1 channel
                ## and flipped from Blender's convention
                if sel == 'depth':
                    out = out[0,:,:]
                    out = 1 - out
                ## normals are 3 channels
                ## scaled from [0,1] --> [-1,1] 
                if sel == 'normals':
                    out = image_to_vector(out)
            outputs.append(out)
        return outputs

    def __len__(self):
        files = self.data_files[0]
        if type(files) == np.ndarray:
            return files.shape[0]
        elif type(files) == dict:
            return len(files['reflectance'])
        else:
            return len(files)


if __name__ == '__main__':
    import time
    directory = '../../dataset/output/'
    datasets = 'car_normalized,boat_normalized'
    selections = ['input', 'mask', 'albedo', 'normals', 'lights']
    # selection_fn = lambda fname: 'shading' in fname
    dset = IntrinsicDataset(directory, datasets, selections)
    loader = torch.utils.data.DataLoader(dset, batch_size=32, num_workers=4)
    print ('done init')
    time_0 = time.time()
    for i, inp in enumerate(loader):
        print (i, [t.size() for t in inp], len(inp))
    print ('total time: ', time.time() - time_0)

