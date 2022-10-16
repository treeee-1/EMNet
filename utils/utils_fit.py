import os

import torch
from nets.deeplabv3_training import CE_Loss,Focal_Loss,cross_entropy_loss_RCF
from tqdm import tqdm

from utils.utils import get_lr



def mutifit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,  cls_weights, num_classes, save_period, save_dir):
    total_loss      = 0
    val_loss   = 0


    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: 
                break
            imgs, pngs1,pngs2= batch

            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)

                pngs1    = torch.from_numpy(pngs1).long()
                pngs2    = torch.from_numpy(pngs2).long()
                weights = torch.from_numpy(cls_weights)

                if cuda:
                    imgs    = imgs.cuda()

                    pngs1    = pngs1.cuda()
                    pngs2    = pngs2.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs,pngs2)

            loss1 = CE_Loss(outputs[0], pngs1, weights, num_classes = num_classes)

            loss2 = torch.zeros(1).cuda()

            loss2 =  cross_entropy_loss_RCF(outputs[1], pngs2)
            loss = loss1 + loss2*0.000001

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            loss2      += loss2.item()
            
            pbar.set_postfix(**{  'loss2': loss2,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            
            imgs, pngs1,pngs2= batch

            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)

                pngs1    = torch.from_numpy(pngs1).long()
                pngs2    = torch.from_numpy(pngs2).long()
                weights = torch.from_numpy(cls_weights)

                if cuda:
                    imgs    = imgs.cuda()

                    pngs1    = pngs1.cuda()
                    pngs2    = pngs2.cuda()
                    weights = weights.cuda()

                outputs = model_train(imgs,pngs2)

                loss1 = CE_Loss(outputs[0], pngs1, weights, num_classes = num_classes)

                loss2 = torch.zeros(1).cuda()
 
                loss2 =  cross_entropy_loss_RCF(outputs[1], pngs2)
                loss = loss1 + loss2*0.0001

                val_loss    += loss.item()

                
            pbar.set_postfix(**{  'loss2': loss2,
                                                        'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f & loss2:%.3f' % (total_loss / epoch_step, val_loss / epoch_step_val, loss2/epoch_step))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
