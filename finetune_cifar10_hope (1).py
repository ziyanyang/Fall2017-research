import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.datasets import CIFAR10


import hypercolumn_alexnet as ha
import compact_hypercolumn_alexnet as cha
import smaller_compact_hypercolumn_alexnet as scha
import binary_smaller_compact_hypercolumn_alexnet as bscha
import hypercolumn_bn_alexnet as hba
import binary_hypercolumn_bn_alexnet as bhba

import binarizer_classifier_k as bck
import squeezer_binarizer_classifier as sbc
import numpy as np
from scipy import linalg 
from numpy import linalg as LA
from numpy.matlib import rand,zeros,ones,empty,eye

parser = argparse.ArgumentParser(description = 'PyTorch ImageNet Training')
parser.add_argument('data', metavar = 'DIR',
                    help = 'path to dataset')
parser.add_argument('--model', default = 'halexnet', type = str,
                    help = 'halexnet | hbalexnet')
parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                    help = 'number of data loading workers (default: 4)')
parser.add_argument('--epochs', default = 90, type = int, metavar = 'N',
                    help = 'number of total epochs to run')
parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N',
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default = 256, type = int,
                    metavar = 'N', help = 'mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default = 0.01, type = float,
                    metavar = 'LR', help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                    help = 'momentum')
parser.add_argument('--weight-decay', '--wd', default = 1e-4, type = float,
                    metavar = 'W', help = 'weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default = 10, type = int,
                    metavar = 'N', help = 'print frequency (default: 10)')
parser.add_argument('--resume', default = '', type = str, metavar = 'PATH',
                    help = 'path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest = 'evaluate', action = 'store_true',
                    help = 'evaluate model on validation set')
parser.add_argument('--pretrained', dest = 'pretrained', action = 'store_true',
                    help = 'use pre-trained model')
parser.add_argument('--custom-schedule', action = 'store_true',
                    help = 'use customly defined learning rate schedule')
parser.add_argument('--logDir', default = 'logs',
                    help = 'directory to store logs')
parser.add_argument('--pretrainedPath', default = 'fixed_logsSCHAlexnet/model_best.pth.tar',
                    help = 'directory to store logs')
parser.add_argument('--optimizer', default = 'sgd',
                    help = 'optimizer to use (default: sgd), but also could be adam')
parser.add_argument('--k', '--kvalue', default = 128, type = int,
                    metavar = 'K', help = 'initial k value')

parser.add_argument('--use_pca_reg', default = False, help = 'use pca loss')
parser.add_argument('--use_itq_reg', default = False, help = 'use itq loss')

parser.add_argument('--hr', '--weight_hope_regularization', default = 0.1, type = float,
                    metavar = 'hr', help = 'initial hope learning rate')

best_prec1 = 0

def main():
    global args, best_prec1, rotation, k, train_mean, train_std, pca_matrix
    global cca_rotation, cca_train_mean, cca_train_std, cca_matrix
    args = parser.parse_args()

    print("pretrained: ",args.pretrained)

    # create model
    if args.model == 'halexnet':
        model = ha.hypercolumn_alexnet(pretrained = args.pretrained, num_classes = 1000)
    elif args.model == 'chalexnet':
        model = cha.compact_hypercolumn_alexnet(pretrained = args.pretrained, num_classes = 1000)
    elif args.model == 'schalexnet':
        model = scha.smaller_compact_hypercolumn_alexnet(pretrained = args.pretrained, num_classes = 10)
        pre_model = None
    elif args.model == 'bschalexnet':
        model = bscha.binary_smaller_compact_hypercolumn_alexnet(num_classes = 1000, 
                                                                 model_path = args.pretrainedPath)
    elif args.model == 'hbalexnet':
        model = hba.hypercolumn_bn_alexnet(pretrained = args.pretrained, num_classes = 1000)
    elif args.model == 'bhbalexnet':
        model = bhba.binary_hypercolumn_bn_alexnet(pretrained = args.pretrained, num_classes = 1000)
    elif args.model == 'itq' or args.model == 'cca':
        k = args.k
        pre_model = scha.smaller_compact_hypercolumn_alexnet(num_classes = 10)
        print('loading pre-trained weights..')
        checkpoint = torch.load(args.pretrainedPath)
        pre_model.load_state_dict(checkpoint['state_dict'])
        print('...pretrained weights loaded')
        
        pre_model = pre_model.cuda()
        pre_model.eval()
        # model is only binarizer + classifier
        model = bck.binarizer_classifier_k(num_classes = 10, k_value = k)
        
    elif args.model == 'smp':
        k = args.k
        pre_model = scha.smaller_compact_hypercolumn_alexnet(num_classes = 1000)
        print('loading pre-trained weights..')
        checkpoint = torch.load(args.pretrainedPath)
        pre_model.load_state_dict(checkpoint['state_dict'])
        print('...pretrained weights loaded')
        
        pre_model = pre_model.cuda()
        pre_model.eval()
        #model = sbc.squeezer_binarizer_classifier(num_classes = 1000, k_value = k, use_pca_reg = args.use_pca_reg,use_itq_reg = args.use_itq_reg)
        # with pre-trained itq,pca matrices and post models
        # model = sbc.squeezer_binarizer_classifier(num_classes = 1000, k_value = k, pretrained = True) 
        model = sbc.squeezer_binarizer_classifier(num_classes = 1000, k_value = k, use_pca_reg = args.use_pca_reg)
        
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum = args.momentum,
                                    weight_decay = args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        trainF = open(os.path.join(args.logDir, 'train.csv'), 'a')
        accuracyF = open(os.path.join(args.logDir, 'accuracy.csv'), 'a')
        testF = open(os.path.join(args.logDir, 'test.csv'), 'a')
        OrthLoss = open(os.path.join(args.logDir, 'orth_loss.csv'), 'a')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        trainF = open(os.path.join(args.logDir, 'train.csv'), 'w')
        accuracyF = open(os.path.join(args.logDir, 'accuracy.csv'), 'w')
        testF = open(os.path.join(args.logDir, 'test.csv'), 'w')
 

    cudnn.benchmark = True

    # Data loading code: CIFAR10
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers)
    """
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                        (0.2023, 0.1994, 0.2010))])
    trainset = CIFAR10(root=args.data, train = True, transform = imgTransform, download=True)
    valset = CIFAR10(root=args.data, train = False, transform = imgTransform, download=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, 
                                          shuffle = True, num_workers = args.workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, 
                                        shuffle = False, num_workers = args.workers)
    """
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    # pick a random element from training data and get rotation matrix R
    if args.model == 'itq':
        
            dataiter = iter(train_loader)
            for i in range(0,40):
                sample, labels = dataiter.next()
                _, sample, _= pre_model(torch.autograd.Variable(sample, volatile=True).cuda())

                if i == 0:
                    X_sample = sample
                else:
                    X_sample = torch.cat((X_sample, sample), 0)
                    
            print(X_sample.data.shape)
            
            # 10240 = batchsize * 40
            X_sample = X_sample.data.permute(0,2,3,1).contiguous().view(10240*6*6,128).cpu().numpy() 
            
            X_sample, train_mean, train_std = get_mean_std(X_sample)
            print("train_mean is found, size is: ", train_mean.shape)
            print("train_std is found, size is:", train_std.shape)
            pca_encode_train, pca_matrix = PCA_encoding(X_sample)
            rotation = ITQ_encoding(pca_encode_train,50)[1]
            print("rotation is found, shape is: ", rotation.shape)
            # store rotation, pca_matrix, train_mean, train_std
          
            np.save(os.path.join(args.logDir, 'rotation'), rotation)
            np.save(os.path.join(args.logDir, 'pca_matrix'), pca_matrix)
            np.save(os.path.join(args.logDir, 'train_mean'), train_mean)
            np.save(os.path.join(args.logDir, 'train_std'), train_std)
            
    elif args.model == 'cca':
        
        dataiter = iter(train_loader)
        print("CCA training...")
        """
        X = [[1,2,3,4,5,6],[7,8,9,8,7,6],[3,4,5,6,7,5],[3,4,5,6,5,4],[3,4,5,6,7,8]]
        Y = [[3,4,6,3,5],[4,5,6,4,3],[3,4,5,6,4],[4,5,6,7,7],[2,3,4,5,7]]
        Z = np.column_stack((X, Y))
        test_vec = CCA_encoding(X, Z, np.size(X,1), np.size(Y,1))
        print(test_vec)
        return
        """
        for i in range(0, 40):
            sample, labels = dataiter.next()
            _, sample, _ = pre_model(torch.autograd.Variable(sample, volatile=True).cuda())
            
            
            if i == 0:
                X_sample = sample
                X_label = labels
                #print('label is {}, with type{}'.format(X_label, type(X_label)))
                #X_sample = torch.cat((X_label, labels),0)
            else:
                X_sample = torch.cat((X_sample, sample), 0)
                X_label = torch.cat((X_label, labels), 0)
        #print(type(X_sample))
        # 40- 10240, 1-256
        X_sample = X_sample.data.permute(0,2,3,1).contiguous().view(10240*6*6,128).cpu().numpy() 
        X_sample, cca_train_mean, cca_train_std = get_mean_std(X_sample)
        X_label = X_label.cpu().numpy()
        X_label = np.repeat(X_label, 36)
        #print('we get data shape {}'.format(X_sample.shape))
        #print('label {} shape{},'.format(X_label, X_label.shape))
        X_labels = np.zeros((10240*6*6, 10), dtype=int)
        i = 0
        for l in X_labels:
            l[X_label[i]] = 1
            i+= 1
        #print(X_labels[0])
        combined_X_sample = np.column_stack((X_sample, X_labels))
        #9216, 138
        print('we get new data shape {}'.format(combined_X_sample.shape))
        
        cx = np.size(X_sample,1)
        cy = np.size(X_labels,1)
        #print('cx should be 128 {} and xy should be 10 {}'.format(cx,cy))
        
        cca_matrix,cca_encode_train = CCA_encoding(X_sample, combined_X_sample, cx, cy)
        
        print("CCA matrix is found with shape {} and resulted matrix shape is {}".format(cca_matrix.shape, cca_encode_train.shape))
        
        cca_rotation = ITQ_encoding(cca_encode_train,50)[1]
        print(cca_rotation.shape)
        print("cca rotation is found.")
        # store rotation, pca_matrix, train_mean, train_std
        
        np.save(os.path.join(args.logDir, 'cca_rotation'), cca_rotation)
        np.save(os.path.join(args.logDir, 'cca_matrix'), cca_matrix)
        np.save(os.path.join(args.logDir, 'cca_train_mean'), cca_train_mean)
        np.save(os.path.join(args.logDir, 'cca_train_std'), cca_train_std)


    for epoch in range(args.start_epoch, args.epochs):
        if args.custom_schedule:
            custom_adjust_learning_rate(optimizer, epoch)
        else:
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(args, epoch, train_loader, pre_model, model, criterion, optimizer, trainF, accuracyF)

        # evaluate on validation set
        prec1 = validate(args, epoch, val_loader, pre_model, model, criterion, testF, accuracyF)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join(args.logDir, 'checkpoint.pth.tar'))
        os.system('python plot.py {} &'.format(args.logDir))

    trainF.close()
    testF.close()
    accuracyF.close()


def train(args, epoch, train_loader, pre_model, model, criterion, optimizer, trainF, accuracyF):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    hopes = AverageMeter()
    itqs = AverageMeter()
    pcas = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        
        
        if args.model == 'itq':
            _, pre_output, _= pre_model(input_var)
            R = rotation
            batch_results = []
            for com_hyper in pre_output:              
                com_hyper = com_hyper.data.permute(1,2,0).contiguous().view(6*6,128) 
                com_hyper = center_data(com_hyper.cpu().numpy(), train_mean, train_std)
                XW = np.dot(com_hyper, pca_matrix)
                XW = np.dot(XW, R)
                XW = XW.real.astype(float)
                XW = XW.T # get k*36
                XW = XW.reshape(k, 6, 6) #k
                batch_results.append(XW)
            batch_results = np.array(batch_results)
            batch_results = torch.from_numpy(batch_results).float()
            batch_results = torch.autograd.Variable(batch_results).cuda()
            output = model(batch_results)
            loss = criterion(output, target_var)
            
        elif args.model == 'smp':
            _, pre_output, _= pre_model(input_var)
            output, hope, itq, pca = model(pre_output)
            
            loss = criterion(output, target_var)
            
            whr = args.hr
            """
            loss = loss + whr*hope
            hopes.update(whr*hope.data[0], input.size(0))
            itqs.update(itq.data[0], input.size(0))
            pcas.update(pca.data[0], input.size(0))
            """
            loss = loss + whr *  1 * hope + itq + (1-whr) *  1 * pca
            hopes.update(whr *  1 * hope.data[0], input.size(0))
            itqs.update(itq.data[0], input.size(0))
            pcas.update((1-whr) *  1*pca.data[0], input.size(0))
            """
            loss = loss + whr * hope + 0.5*(1-whr)*0.01*itq + 0.5*(1-whr)*pca
            hopes.update(whr * hope.data[0], input.size(0))
            itqs.update(0.5*(1-whr)*0.01*itq.data[0], input.size(0))
            pcas.update(0.5*(1-whr)*pca.data[0], input.size(0))
            """
            
        elif args.model == 'cca':
            _, pre_output, _= pre_model(input_var)
            R = cca_rotation
            batch_results = []
            for com_hyper in pre_output:              
                com_hyper = com_hyper.data.permute(1,2,0).contiguous().view(6*6,128) 
                com_hyper = center_data(com_hyper.cpu().numpy(), cca_train_mean, cca_train_std)
                XW = np.dot(com_hyper, cca_matrix)
                XW = np.dot(XW, R)
                XW = XW.real.astype(float)
                XW = XW.T # get k*36
                XW = XW.reshape(k, 6, 6) #k
                batch_results.append(XW)
            batch_results = np.array(batch_results)
            batch_results = torch.from_numpy(batch_results).float()
            batch_results = torch.autograd.Variable(batch_results).cuda()
            output = model(batch_results)
            loss = criterion(output, target_var)
            
        else:
            output, _, _ = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk = (1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """
        if args.model == 'smp':
            model.hope_normalization()
        """
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        trainF.write('{},{},{}\n'.format(epoch, losses.avg, top1.avg))
        trainF.flush()

        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Hope {hope.val:.4f} ({hope.avg:.4f})\t'
                  'Itq {itq.val:.4f} ({itq.avg:.4f})\t'
                  'Pca {pca.val:.4f} ({pca.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time = batch_time,
                   data_time = data_time, loss = losses, hope = hopes,itq = itqs,pca = pcas, top1 = top1, top5 = top5))

    accuracyF.write('{},{},{},{},{},{},{},{},{}\n'.format(
        epoch, 'train', top1.avg, top5.avg, losses.avg, hopes.avg, itqs.avg,pcas.avg,losses.avg + hopes.avg + itqs.avg + pcas.avg))
    accuracyF.flush()


def validate(args, epoch, val_loader, pre_model, model, criterion, testF, accuracyF):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile = True).cuda()
        target_var = torch.autograd.Variable(target, volatile = True).cuda()

        # compute output
        if args.model == 'itq':
            _, pre_output, _= pre_model(input_var)
            R = rotation
            batch_results = []
            for com_hyper in pre_output:              
                com_hyper = com_hyper.data.permute(1,2,0).contiguous().view(6*6,128) 
                com_hyper = center_data(com_hyper.cpu().numpy(), train_mean, train_std)
                XW = np.dot(com_hyper, pca_matrix)
                XW = np.dot(XW, R)
                XW = XW.real.astype(float)
                XW = XW.T # get k*36
                XW = XW.reshape(k, 6, 6) #k
                batch_results.append(XW)
            batch_results = np.array(batch_results)
            batch_results = torch.from_numpy(batch_results).float()
            batch_results = torch.autograd.Variable(batch_results).cuda()
            output = model(batch_results)
            loss = criterion(output, target_var)
            
        elif args.model == 'smp':
            _, pre_output, _= pre_model(input_var)
            output, hope, itq, pca = model(pre_output)
            loss = criterion(output, target_var)
            
        elif args.model == 'cca':
            _, pre_output, _= pre_model(input_var)
            R = cca_rotation
            batch_results = []
            for com_hyper in pre_output:              
                com_hyper = com_hyper.data.permute(1,2,0).contiguous().view(6*6,128) 
                com_hyper = center_data(com_hyper.cpu().numpy(), cca_train_mean, cca_train_std)
                XW = np.dot(com_hyper, cca_matrix)
                XW = np.dot(XW, R)
                XW = XW.real.astype(float)
                XW = XW.T # get k*36
                XW = XW.reshape(k, 6, 6) #k
                batch_results.append(XW)
            batch_results = np.array(batch_results)
            batch_results = torch.from_numpy(batch_results).float()
            batch_results = torch.autograd.Variable(batch_results).cuda()
            output = model(batch_results)
            loss = criterion(output, target_var)
        else:
            output, _, _ = model(input_var)
            loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk = (1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time = batch_time, loss = losses,
                   top1 = top1, top5 = top5))

    testF.write('{},{},{}\n'.format(epoch, losses.avg, top1.avg))
    testF.flush()
    accuracyF.write('{},{},{}\n'.format(epoch, 'test', top1.avg))
    accuracyF.flush()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1 = top1, top5 = top5))

    return top1.avg

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.logDir, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def custom_adjust_learning_rate(optimizer, epoch):
    if epoch >= 0 and epoch < 30:
        lr = args.lr
    elif epoch >= 30 and epoch < 34:
        lr = args.lr * 0.1
    elif epoch >= 34 and epoch < 38:
        lr = args.lr * 0.01
    elif epoch >=38 and epoch < 42:
        lr = args.lr * 0.001
    elif epoch >=42:
        lr = args.lr * 0.0001

    print('Starting new LR', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def ITQ_encoding(V, iteration):
    c = np.shape(V)[1]
    R = np.random.rand(c,c)
    U, s, V2 = linalg.svd(R, lapack_driver = 'gesvd')
    R = U
    for i in range(0, iteration):
        # update B
        Z = np.dot(V, R)
        B = np.sign(Z)  # 1 or -1

        # update R
        C = np.dot(np.transpose(B),V)
        U, s, V2 = linalg.svd(C, lapack_driver = 'gesvd')
        V2 = V2.T
        U = U.T
        R = np.dot(V2, U)
    # make B binary, so 0 or 1
    B = (B+1)/2
    
    return B, R




def PCA_encoding(X):
    X_cov = np.cov(X, rowvar = False)

    # get most significant k eigenvectors of covariance matrix
    evals,evecs = LA.eig(X_cov)
    EvalsVecs = sorted(zip(evals,evecs.T), key=lambda x: x[0].real, reverse=True)
    EvalsVecs = EvalsVecs[:k] #k
    evecs = [list(x[1]) for x in EvalsVecs]

    # get the input for ITQ
    evecs = np.array(evecs)
    XW = np.dot(X, evecs.T)
    #print("XW:")
    return XW, evecs.T

def get_mean_std(data_train):
    #n_samples = data_m.shape[0]  
    # data = copy.deepcopy(data_train)
    trainmean = np.mean(data_train, axis=0)
    trainstd = np.std(data_train, axis = 0, ddof = 1)
    data_train -= trainmean
    data_train /= trainstd
    return data_train, trainmean, trainstd
    
def center_data(data_test, mean, std):
    data_test -= mean
    data_test /= std
    return data_test

def CCA_encoding(X, Z, sx, sy):
    
    C = np.cov(Z, rowvar = False)
    reg = 0.0001
    Cxx = C[0:sx:1, 0:sx:1]
    Cxx = Cxx + reg*eye(sx)
    Cxy = C[0:sx:1, sx:sx+sy:1]
    Cyx = Cxy.T
    Cyy = C[sx:sx+sy:1,sx:sx+sy:1]+reg*eye(sy)
    Rx = LA.cholesky(Cxx).T
    invRx = LA.inv(Rx)
    Z = np.dot(invRx.T,Cxy)
    Z = np.dot(Z, LA.lstsq(Cyy,Cyx)[0])
    Z = np.dot(Z,invRx)
    Z = Z.T+Z
    Z = 0.5*Z
    
    
    r,Wx = LA.eig(Z)
    """
    Wx = np.array([[0.0035,-0.0013,-0.5898,-0.6015,-0.2972,-0.4494],[-0.9859,-0.1675,-0.0019,-0.0019,-0.0010,-0.0015],
          [-0.1675,0.9859,-0.0011,-0.0011,-0.0006,-0.0008],[0.0000,-0.0000,-0.3627,0.7052,-0.6056,-0.0674],
          [-0.0000,0.0000,0.1795,-0.3726,-0.6160,0.6705],[-0.0000,0.0000,-0.6989,0.0460,0.4069,0.5865]])
    r = np.array([0, 0, 0.9973,0.9989,0.9997,0.9999])
    """
    r = np.nan_to_num(np.sqrt(r.real))
    Wx = np.dot(invRx,Wx)
    #print('r should be a single array,size {} and is {} '.format(r.shape, r))
    #print('Wx should match{} '.format(Wx))
    
    res = sorted(zip(r,Wx.T), key=lambda x: x[0].real, reverse=True)
    res = res[:k]
    Wx = [list(x[1]) for x in res]
    Wx = np.array(Wx)
    r = [x[0] for x in res]
    #print('sorted Wx {}, r {}'.format(Wx, r))
    
    eigenvector = np.dot(Wx.T,np.diag(r))
    eigenvector = np.squeeze(eigenvector)
    #print('eigenvector {}'.format(eigenvector))
    
    E = np.dot(X,eigenvector)
    
    
    return eigenvector, E
    

if __name__ == '__main__':
    main()

