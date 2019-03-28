import sys, os, argparse, torch, pdb, matplotlib, scipy.misc, numpy
sys.path.insert(0,sys.path[0] + '/intrinsics_network')
print(sys.path)
import models, pipeline
from torch.autograd import Variable
PATH ="intrinsics_network/saved/decomposer/state.t7"
def detachAndSqueeze(result):
        result = torch.squeeze(result)
        result = result.permute(1,2,0)
        result = result.data
        return result
def doIntrinsicDecomposition(input,mask):
    input = input[None,:,:,:]
    mask = mask[None,:,:,:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.Decomposer().cuda()
    model.load_state_dict(torch.load(PATH))

    input_tensor = torch.from_numpy(numpy.copy(input))
    input_tensor = input_tensor.permute(0,3,2,1)
    input_tensor = input_tensor.permute(0,1,3,2)
    input_tensor = input_tensor.float().to(device)
  

    mask_tensor = torch.from_numpy(numpy.copy(mask))
    mask_tensor = mask_tensor.permute(0,3,2,1)
    mask_tensor = mask_tensor.permute(0,1,3,2)
    mask_tensor = mask_tensor.float().to(device)

    
    refl_pred, depth_pred, shape_pred, lights_pred = model.forward(input_tensor, mask_tensor)
    refl_pred = detachAndSqueeze(refl_pred)
    shape_pred = detachAndSqueeze(shape_pred)  
    return refl_pred


