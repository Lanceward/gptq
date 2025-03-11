import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, Weight = None
    ):
        if Weight != None:
            self.layer.weight.data = Weight
        W = self.layer.weight.data.clone()
                    
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H.clone()
        #del self.H
        if DEBUG:
            print(H.shape)
        dead = torch.diag(H) == 0 # find weights that has no local curvature, like x^4
        H[dead, dead] = 1 # set the diagnal of these weights to 1(?)
        W[:, dead] = 0 # set colums of these weights to 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True) # sort diagnal Hessian in descending. We want to deal with large errors first. More chance for other weights to adjust
            W = W[:, perm] # Rearrange columns of W by the permutation
            H = H[perm][:, perm] # Rearrange H by the permutation. Note both rows and columns needs rearrangement
            invperm = torch.argsort(perm) # get the permutation that will return original W

        Losses = torch.zeros_like(W) # Loss matrix
        Q = torch.zeros_like(W) # quantized weights(not compressed)

        damp = percdamp * torch.mean(torch.diag(H)) # get dampening value as mentioned in page 5
        diag = torch.arange(self.columns, device=self.dev) # vector [0, 1, ... columns-1]
        H[diag, diag] += damp # add dampening to diagnal values for stability (?)
        H = torch.linalg.cholesky(H) # compute cholesky of H: L
        H = torch.cholesky_inverse(H) # compute H^-1 from L
        H = torch.linalg.cholesky(H, upper=True) # compute cholesky of H^-1, L^T
        Hinv = H

        print("ERROR BEFORE QUANT: ", torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        al = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1 # count of columns in block

            W1 = W[:, i1:i2].clone() # get weight in block
            Q1 = torch.zeros_like(W1) # initialize quant weight for block
            Err1 = torch.zeros_like(W1) #
            Losses1 = torch.zeros_like(W1) #
            Hinv1 = Hinv[i1:i2, i1:i2] # get H^-1 of block

            for i in range(count):
                w = W1[:, i] # ith column in block
                d = Hinv1[i, i] # diag of hessian

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten() # quantize column w
                Q1[:, i] = q # store quantized column in Q1
                Losses1[:, i] = (w - q) ** 2 / d ** 2 # calculate losse of ith column

                err1 = (w - q) / d # Quantization error
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) # update all remaining weight in block
                Err1[:, i] = err1 # record error

            Q[:, i1:i2] = Q1 # put updates in this block to layer
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) # update all remaining weight in layer

            if DEBUG:
                if isinstance(self.layer, transformers.Conv1D):
                    if actorder:
                        qidx = perm[:i2]
                        widx = perm[i2:]
                        Q_t = Q[:, invperm]
                        W_t = W[:, invperm]
                    else:
                        qidx = torch.arange(0, i2, device=self.dev)
                        widx = torch.arange(i2, self.rows, device=self.dev)
                        Q_t = Q
                        W_t = W
                    
                    self.layer.weight.data[qidx, :] = Q_t[:, qidx].t()
                    self.layer.weight.data[widx, :] = W_t[:, widx].t()
                #activation loss
                al.append(torch.sum((self.layer(self.inp1) - self.out1) ** 2).item())
                print("activa loss: ", al[-1])
                #print("weight loss: ", torch.sum(Losses))

        #torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        
        return al

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
