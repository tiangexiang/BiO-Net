from torch.utils.data import DataLoader
import sys
import os
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from tqdm import tqdm
from .metrics import *
from .dataloader import BiONetDataset
from .model import BiONet


device = torch.device('cuda:0')

def train(args):
    # augmentations
    transforms = iaa.Sequential([
            iaa.Rotate((-15., 15.)),
            iaa.TranslateX(percent=(-0.05,0.05)),
            iaa.TranslateY(percent=(-0.05,0.05)),
            iaa.Affine(shear=(-50, 50)),
            iaa.Affine(scale=(0.8, 1.2)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])

    # load data and create data loaders
    train_set = BiONetDataset(args.train_data, 'monuseg', batchsize=args.batch_size, steps=args.steps, transforms=transforms)
    test_set = BiONetDataset(args.valid_data, args.valid_dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # create model
    model = BiONet(iterations=args.iter,
                   num_classes=args.num_class,
                   num_layers=4,
                   multiplier=args.multiplier,
                   integrate=args.integrate).to(device).float()

    criterion = BCELoss()
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)

    # keras lr decay equivalent
    fcn = lambda step: 1./(1. + args.lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    print('model successfully built and compiled.')
  
    if not os.path.isdir("checkpoints/"+args.exp):
    	os.mkdir("checkpoints/"+args.exp)

    best_iou = 0.
    print('\nStart training...')
    for epoch in range(args.epochs):
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.

        # training
        model.train()
        for step, (x, y) in enumerate(tqdm(train_loader, desc='[TRAIN] Epoch '+str(epoch+1)+'/'+str(args.epochs))):
            if step >= args.steps:
                break
            x = x.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(x)

            # loss
            l = criterion(output, y)
            tot_loss += l.item()
            l.backward()
            optimizer.step()

            # metrics
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            iou_score = iou(y, x)
            dice_score = dice_coef(y, x)
            tot_iou += iou_score
            tot_dice += dice_score

            scheduler.step()

        print('[TRAIN] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
              'loss:', tot_loss/args.steps,
              'iou:', tot_iou/args.steps,
              'dice:', tot_dice/args.steps)

        # validation
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(tqdm(test_loader, desc='[VAL] Epoch '+str(epoch+1)+'/'+str(args.epochs))):
                x = x.to(device).float()
                y = y.to(device).float()

                output = model(x)

                # loss
                l = criterion(output, y)
                val_loss += l.item()

                # metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                iou_score = iou(y, x)
                dice_score = dice_coef(y, x)
                val_iou += iou_score
                val_dice += dice_score

        if val_iou/len(test_loader) > best_iou:
            best_iou = val_iou/len(test_loader)
            save_model(args, model)
        
        print('[VAL] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
              'val_loss:', val_loss/len(test_loader),
              'val_iou:', val_iou/len(test_loader),
              'val_dice:', val_dice/len(test_loader),
              'best val_iou:', best_iou)

    print('\nTraining fininshed!')

def evaluate(args):
    # load data and create data loader
    test_set = BiONetDataset(args.valid_data, args.valid_dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if args.model_path is None:
        integrate = '_int' if args.integrate else ''
        weights = '_weights'
        cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+integrate+'_best'+weights+'.pt'
        model_path = "checkpoints/"+args.exp+"/"+cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: '+model_path)
    model = BiONet(iterations=args.iter,
                   num_classes=args.num_class,
                   num_layers=4,
                   multiplier=args.multiplier,
                   integrate=args.integrate).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    criterion = BCELoss()

    val_loss = 0.
    val_iou = 0.
    val_dice = 0.

    segmentations = []

    # validation
    print('\nStart evaluation...')
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(test_loader)):
 
            x = x.to(device).float()
            y = y.to(device).float()

            output = model(x)

            # loss
            l = criterion(output, y)
            val_loss += l.item()

            # metrics
            x, y = output.detach().cpu().numpy(), y.cpu().numpy()
            iou_score = iou(y, x)
            dice_score = dice_coef(y, x)
            val_iou += iou_score
            val_dice += dice_score

            if args.save_result:
                segmentations.append(x)
            
    val_loss = val_loss/len(test_loader)
    val_iou = val_iou/len(test_loader)
    val_dice = val_dice/len(test_loader)
    print('Validation loss:\t', val_loss)
    print('Validation  iou:\t', val_iou)
    print('Validation dice:\t', val_dice)

    print('\nEvaluation finished!')

    if args.save_result:

        # save metrics
        if not os.path.exists("checkpoints/"+args.exp+"/outputs"):
            os.mkdir("checkpoints/"+args.exp+"/outputs")

        with open("checkpoints/"+args.exp+"/outputs/result.txt", 'w+') as f:
            f.write('Validation loss:\t'+str(val_loss)+'\n')
            f.write('Validation  iou:\t'+str(val_iou)+'\n')
            f.write('Validation dice:\t'+str(val_dice)+'\n')
        
        print('Metrics have been saved to:', "checkpoints/"+args.exp+"/outputs/result.txt")

        # save segmentations
        results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
        results = (results > 0.5).astype(np.float32) # Binarization. Comment out this line if you don't want to

        print('Saving segmentations...')
        if not os.path.exists("checkpoints/"+args.exp+"/outputs/segmentations"):
            os.mkdir("checkpoints/"+args.exp+"/outputs/segmentations")

        for i in range(results.shape[0]):
            plt.imsave("checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+".png",results[i,:,:,0],cmap='gray') # binary segmenation

        print('A total of '+str(results.shape[0])+' segmentation results have been saved to:', "checkpoints/"+args.exp+"/outputs/segmentations/")

def save_model(args, model):
    integrate = '_int' if args.integrate else ''
    weights = '_weights'
    cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+integrate+'_best'+weights+'.pt'
    torch.save({'state_dict':model.state_dict()}, "checkpoints/"+args.exp+"/"+cpt_name)
