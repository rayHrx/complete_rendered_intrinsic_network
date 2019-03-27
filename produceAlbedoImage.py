import sys, os, argparse, torch, pdb, matplotlib, scipy.misc, numpy
import models, pipeline
from torch.autograd import Variable

def detachAndSqueeze(result):
        result = result.data[0]
        result = torch.squeeze(result)
        return result

#python produceAlbedoImage.py --saved_path saved/decomposer/ --state_name state.t7
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default='dataset/',
        help='base path for datasets')  
parser.add_argument('--inference_sets',      type=str,   default='input',  
        help='the folder where images need to be inferenced are')      
parser.add_argument('--num_pred',      type=int,   default=1,  
        help='the number of images to be predicted')         
parser.add_argument('--state_name',      type=str,   default='state.t7',  
        help='default state to load')
parser.add_argument('--inference',     type=list,  default=['composite', 'mask'],
        help='intrinsic images to load from the train and val sets')
parser.add_argument('--saved_path',      type=str,   default='saved/decomposer/',  
        help='the folder where saved weights are')
parser.add_argument('--loaders',    type=int,   default=4,
        help='number of parallel data loading processes')
parser.add_argument('--output_path',    type=str,   default='dataset/input/predict/',
        help='number of parallel data loading processes')
args = parser.parse_args()
PATH = args.saved_path + args.state_name
model = models.Decomposer().cuda()
model.load_state_dict(torch.load(PATH))
#train_set = pipeline.IntrinsicDataset(args.data_path, args.inference_sets, args.inference, array=args.array, size_per_dataset=args.num_train)

inference_sets = pipeline.InferenceDataset(args.data_path, args.inference_sets, args.inference, size_per_dataset = args.num_pred)
inference_loader = torch.utils.data.DataLoader(inference_sets, batch_size=1, num_workers=args.loaders)

index = 0
for ind, tensors in enumerate(inference_loader):
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        print("Tensor shaaape is : ",tensors)
        inp, mask = tensors
        print("Input shaaape is : ",inp.shape)
        print("Mask shaaape is : ",inp.shape)
        # refl_pred = 3*256*256 , shape_pred = 3*256*256 , depth_pred = 256*256
        refl_pred, depth_pred, shape_pred, lights_pred = model.forward(inp, mask)
        refl_pred = detachAndSqueeze(refl_pred)
        depth_pred = detachAndSqueeze(depth_pred)
        shape_pred = detachAndSqueeze(shape_pred)    

        refl_result = numpy.zeros((256,256,3))
        shape_result = numpy.zeros((256,256,3))

        for i in range(0,256):
                for j in range(0,256):
                        for k in range(0,3):
                                refl_result[i][j][k] = refl_pred[k][i][j]
                                shape_result[i][j][k] = shape_pred[k][i][j]

        scipy.misc.imsave(args.output_path + str(index) + 'refl.png', refl_result)
        scipy.misc.imsave(args.output_path + str(index) + 'shape.png', shape_result)   
        index = index + 1




         