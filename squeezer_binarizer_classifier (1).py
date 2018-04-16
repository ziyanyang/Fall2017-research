# use pca loss and itq loss: squeezer_binarizer_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
class HardTanh_F(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        return input.sign()

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class HardTanh(nn.Module):
    def forward(self, input):
        return HardTanh_F()(input)
    
class Squeezer_Binarizer_Classifier(nn.Module):

    def __init__(self, num_classes = 1000, k_value = 32, use_pca_reg = False, use_itq_reg = False, pretrained= False):

        super(Squeezer_Binarizer_Classifier, self).__init__()
        print("model pre-trained: {}".format(pretrained))
        self.use_pca_reg = use_pca_reg
        self.use_itq_reg = use_itq_reg
        self.pretrained = pretrained
        
        self.squeezer = nn.Conv2d(128, k_value, kernel_size = 1, bias = False)
        self.quantizer = nn.Conv2d(k_value, k_value, kernel_size = 1, bias = False)
        
        self.binarizer = nn.Sequential(
                          nn.BatchNorm2d(k_value),
                          HardTanh(),
        )

        
        self.classifier = nn.Sequential(
                          nn.Dropout(),
                          nn.Linear(k_value * 6 * 6, 4096),
                          nn.ReLU(inplace=True),
                          nn.Dropout(),
                          nn.Linear(4096, 4096),
                          nn.ReLU(inplace=True),
                          nn.Linear(4096, num_classes),
                        )
    
        mean = np.load('train_mean.npy') # no matter which condition, we actually need this mean data to center our data
        tensor_mean = torch.from_numpy(mean).float()
        self.mean = torch.autograd.Variable(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(tensor_mean, 1), 2),0)).cuda()
        
        # load mean, pca_matrix and itq_matrix
        if self.pretrained:
            
            
            pca = np.load('pca_matrix32.npy')
            itq = np.load('itq_matrix32.npy')
            
            
            pca = torch.from_numpy(pca.T).float()
            pca = torch.unsqueeze(torch.unsqueeze(pca, 2), 3)
            self.squeezer.weight = nn.Parameter(pca)
        
            itq = torch.from_numpy(itq.T).float()
        
            itq = torch.unsqueeze(torch.unsqueeze(itq, 2), 3)
            self.quantizer.weight = nn.Parameter(itq)
            
            print("loaded pre-trained pca and itq matrices.")
            ldict = torch.load('imgnet_itq32_newlog/model_best.pth.tar')
            self.binarizer[0].weight.data.copy_(ldict['state_dict']['binarizer.0.weight'])
            self.binarizer[0].bias.data.copy_(ldict['state_dict']['binarizer.0.bias'])
            self.binarizer[0].running_mean.copy_(ldict['state_dict']['binarizer.0.running_mean'])
            self.binarizer[0].running_var.copy_(ldict['state_dict']['binarizer.0.running_var'])
            self.classifier[1].weight.data.copy_(ldict['state_dict']['classifier.1.weight'])
            self.classifier[1].bias.data.copy_(ldict['state_dict']['classifier.1.bias'])
            self.classifier[4].weight.data.copy_(ldict['state_dict']['classifier.4.weight'])
            self.classifier[4].bias.data.copy_(ldict['state_dict']['classifier.4.bias'])
            self.classifier[6].weight.data.copy_(ldict['state_dict']['classifier.6.weight'])
            self.classifier[6].bias.data.copy_(ldict['state_dict']['classifier.6.bias'])
            print("loaded pre-trained weights.")

    def hope_normalization(self):
        weights = []
        if self.use_pca_reg:
            weights.append(self.squeezer.weight)
            
        if self.use_itq_reg:
            weights.append(self.quantizer.weight)
          
        if self.pretrained:
            # only orthogonal loss
            weights.append(self.squeezer.weight)
            weights.append(self.quantizer.weight)
            
        for weight in weights:
            weight = weight.data.view(weight.size(0), -1)
            norms = weight.norm(2, 1, keepdim = True).expand_as(weight)
            weight.div_(norms)
            
    def center_data(self, x, mean):
        mean = torch.mean(x, 0)
        #mean = self.mean
        centered_x = x.sub(mean)
        return centered_x
    
    def forward(self, x):
        
        weights = []
        
        if self.pretrained:
            weights.append(self.squeezer.weight)
            weights.append(self.quantizer.weight)
        """
        if self.pretrained or self.use_itq_reg or self.use_pca_reg:
            x = x.sub(self.mean)
        """
        bn_x = self.binarizer(self.squeezer(x))
        # Classify into num_classes scores.
       
        predictions = self.classifier(bn_x.view(x.size(0), -1))
        
        hope_term = torch.autograd.Variable(torch.zeros(1).cuda())
        itq_term = torch.autograd.Variable(torch.zeros(1).cuda())
        pca_term = torch.autograd.Variable(torch.zeros(1).cuda())
        
        
        if self.use_itq_reg:
            weights.append(self.quantizer.weight)
            # calculate |sign(XW)-XWR|Frobenius norm
            inputs = x.permute(0,2,3,1).contiguous().view(-1,x.size(1)) 
            sample_n = inputs.size(0)
            #inputs = inputs.sub(self.mean)
            #inputs = self.center_data(inputs, self.mean)
            rparam = self.squeezer.weight.view(self.squeezer.weight.size(0), -1)         
            rotation = self.quantizer.weight.view(self.quantizer.weight.size(0),-1)         
            B = torch.mm(inputs, rparam.t()).sign()
            VR = torch.mm(torch.mm(inputs, rparam.t()),rotation)
            itqloss = B.sub(VR)
            itq_score = torch.pow(itqloss,2).sum()
            itq_score = torch.div(itq_score, sample_n)
            itq_term.add_(itq_score)
            #itq_score = torch.sub()
        if self.use_pca_reg:
            
            weights.append(self.squeezer.weight)   
            # get number of features batch*6*6 as n
            inputs = x.permute(0,2,3,1).contiguous().view(-1,x.size(1))
            #inputs = inputs.sub(self.mean)
            inputs = self.center_data(inputs, self.mean)
            sample_n = inputs.size(0)
            # This part compute trace(W^T*X^T*X*W)/n
            rparam = self.squeezer.weight.view(self.squeezer.weight.size(0), -1)
            pca_score = torch.trace(torch.mm(torch.mm(torch.mm(rparam,inputs.t()),inputs),rparam.t()))
            pca_score = -torch.div(pca_score, sample_n)
            #print("pca score is : {}".format(pca_score))
            pca_term.add_(pca_score)
            
        
            
        for param in weights:
            print(len(weights))
            rparam = param.view(param.size(0), -1)                
            # This part computes |u_i * u_j|
            covmat = torch.mm(rparam, rparam.t())
            acovmat = covmat.abs()
                
            # This part computes |u_i||u_j|
            norms = covmat.diag().sqrt().view(-1, 1)
            norms_ij = torch.mm(norms, norms.t())
                
            # Compute \sum | u_i * u_j | / [|u_i| |u_j|]
            acovmat = torch.div(acovmat, norms_ij)
            hope_score = (acovmat.sum() - acovmat.diag().sum()) / 2
            hope_term.add_(hope_score)
            

        return predictions, hope_term, itq_term, pca_term

def squeezer_binarizer_classifier(**kwargs):
    model = Squeezer_Binarizer_Classifier(**kwargs)
    
    return model



if __name__ == '__main__':
    model = squeezer_binarizer_classifier(num_classes = 1000, k_value = 32, use_pca_reg = False,use_itq_reg = False, pretrained= True)
    model = model.cuda()
    dummy = torch.Tensor(3, 128, 6, 6).normal_(0, 0.01)
    dummy = dummy.cuda()
    
    preds, hopes, itqs, pcas = model(torch.autograd.Variable(dummy))
    #print(preds.size(), hopes.size())
    print(hopes)
    print(itqs)
    print(pcas)

    model.hope_normalization()