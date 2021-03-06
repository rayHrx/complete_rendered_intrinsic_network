import sys, math, numpy as np, pdb
import scipy, os
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pipeline

def detachAndSqueeze(result):
    result = result.data[0]
    result = torch.squeeze(result)
    return result

class DecomposerTrainer:
    def __init__(self, model, loader, lr, lights_mult):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lights_mult = lights_mult



    def __epoch(self):
        self.model.train()
        losses = pipeline.AverageMeter(3)
        progress = tqdm( total=len(self.loader.dataset) )

        for ind, tensors in enumerate(self.loader):
            tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
            inp, mask, refl_targ, depth_targ, shape_targ, lights_targ = tensors

        
            # r_targ = refl_targ.data[0]
            # s_targ = shape_targ.data[0]

            # print("shape 1:",r_targ.shape)
            # print("shape 2:",s_targ.shape)

            # print("shape 1:",r_targ.shape)
            # print("shape 2:",s_targ.shape)

            # refl_result = np.zeros((256,256,3))
            # shape_result = np.zeros((256,256,3))

            # for i in range(0,256):
            #         for j in range(0,256):
            #                 for k in range(0,3):
            #                         refl_result[i][j][k] = r_targ[k][i][j]
            #                         shape_result[i][j][k] = s_targ[k][i][j]

 
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            # print("current path is:",dir_path)
            # scipy.misc.imsave('test_out/refl.png', refl_result)
            # scipy.misc.imsave('test_out/shape.png', shape_result)  

            self.optimizer.zero_grad()
            refl_pred, depth_pred, shape_pred, lights_pred = self.model.forward(inp, mask)
            refl_loss = self.criterion(refl_pred, refl_targ)
            #print("depth_pred------------------------:",depth_pred.shape)
            #print("depth_targ------------------------:",depth_targ.shape)
            # We squeeze the depth_targ to make 256 x 256
            depth_targ = torch.squeeze(depth_targ)
            depth_loss = self.criterion(depth_pred, depth_targ)
            #print("shape_pred------------------------:",shape_pred.shape)
            #print("shape_targ------------------------:",shape_targ.shape)
            shape_loss = self.criterion(shape_pred, shape_targ)
            lights_loss = self.criterion(lights_pred, lights_targ)
            loss = refl_loss + depth_loss + shape_loss + (lights_loss * self.lights_mult)
            loss.backward()
            self.optimizer.step()

            losses.update( [l.item() for l in [refl_loss, shape_loss, lights_loss] ])
            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f | %.5f | %.3f' % (refl_loss.item(), depth_loss.item(), shape_loss.item(), lights_loss.item()) )
        print ('<Train> Losses: ', losses.avgs)
        return losses.avgs

    def train(self):
        # t = trange(iters)
        # for i in t:
        err = self.__epoch()
        # print 
            # t.set_description( str(err) )
        return err

if __name__ == '__main__':
    import sys
    sys.path.append('../')




