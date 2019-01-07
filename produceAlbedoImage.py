import sys, os, argparse, torch, pdb, matplotlib
import models, pipeline
#python produceAlbedoImage.py --saved_path saved/decomposer --state_name state.t7
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default='dataset/output/',
        help='base path for datasets')  
parser.add_argument('--inference_sets',      type=str,   default='input',  
        help='the folder where images need to be inferenced are')               
parser.add_argument('--state_name',      type=str,   default='state.t7',  
        help='default state to load')
parser.add_argument('--inference',     type=list,  default=['composite', 'mask'],
        help='intrinsic images to load from the train and val sets')
parser.add_argument('--saved_path',      type=str,   default='components/test_logger/',  
        help='the folder where saved weights are')
parser.add_argument('--state_name',      type=str,   default='state.t7',  
        help='default state to load')
parser.add_argument('--loaders',    type=int,   default=4,
        help='number of parallel data loading processes')
args = parser.parse_args()
PATH = args.saved_path + state_name
model = models.Decomposer().cuda()
model.laod_State_dict(torch.load(PATH))
#train_set = pipeline.IntrinsicDataset(args.data_path, args.inference_sets, args.inference, array=args.array, size_per_dataset=args.num_train)

inference_sets = pipeline.InferenceLoader(args.data_path, args.inference_sets, args.inference)
inference_loader = torch.utils.data.DataLoader(inference_sets, num_workers=args.loaders)

for ind, tensors in enumerate(self.inference_loader):
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        inp, mask = tensors
        refl_pred, depth_pred, shape_pred, lights_pred = self.model.forward(inp, mask)

matplotlib.image.imsave('dataset/input/predict/refl.png', refl_pred)
matplotlib.image.imsave('dataset/input/predict/depth.png', depth_pred)
matplotlib.image.imsave('dataset/input/predict/shape.png', shape_pred)
matplotlib.image.imsave('dataset/input/predict/lights.png', lights_pred)

         