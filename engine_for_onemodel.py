import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from util_tools.mixup import Mixup
from timm.utils import accuracy, ModelEma
import util_tools.utils as utils
from scipy.special import softmax
from einops import rearrange
import matplotlib
matplotlib.use('Agg')  # or 'Agg' for non-interactive backends
import matplotlib.pyplot as plt
import torch.nn.functional as F

import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE
from loss_functions.triplet_loss import BatchAllTtripletLoss
import time


def cross_train_class_batch(model, samples, target, criterion,criterion2):
    
    output,features,_,_ = model(samples)
    
   
   
    loss=criterion2(output,target)
    
    return loss, output, features



def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module, criterion2,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
  
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 5

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print(f'data_iter_step {data_iter_step}')

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue

        it = start_steps + step  # Global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Move data to GPU
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            print(f"Actual batch size received by mixup_fn: {len(samples)}")
            samples, targets = mixup_fn(samples, targets)

       
        if loss_scaler is None:
            samples = samples.half()
            loss, output, features = cross_train_class_batch(
                model, samples, targets, criterion, criterion2)
        else:
            with torch.cuda.amp.autocast():
                samples = samples.half()
                loss, output, features = cross_train_class_batch(
                    model, samples, targets, criterion, criterion2)
        loss_value = loss.item()
        #print("memory used",torch.cuda.memory_summary())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backpropagation
        if loss_scaler is None:
            loss = loss / update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss = loss / update_freq
            if torch.isnan(loss):
                print("Loss is NaN, exiting...")
                exit()

            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        #torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)

        # Update learning rate & weight decay logs
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        # Log training metrics
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def validation_one_epoch(args, data_loader, model, device):
    criterion=BatchAllTtripletLoss()
    criterion2 = torch.nn.CrossEntropyLoss()


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        target = batch[1]
        print("target:",target)
        batch_size = samples.shape[0]
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output,features,_,_=model(samples)
            # siddhi output,_,_,_ = model(samples)
            
            loss = 0.5*criterion(features, target)+0.5*criterion2(output,target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def predict_with_topk_experts_batch(selector, experts, videos, device, vote='maxlogit'):
    with torch.no_grad():
        B = videos.size(0)
        #selector_logits = selector(videos.to(device))
        selector_output = selector(videos.to(device))
        selector_logits = selector_output if isinstance(selector_output, torch.Tensor) else selector_output[0]# [B, 15]
        top5 = torch.topk(selector_logits, k=1, dim=1).indices  # [B, 5]
        #print("top5 selector logits",  top5)

        final_preds = []
        final_logits=[]
        for i in range(B):
            video = videos[i].unsqueeze(0)  # [1, C, T, H, W]
            top5_experts = top5[i].tolist()

            predictions = []
            logits_list = []
            for k in top5_experts:
                output = experts[k](video.to(device))[0]  # [1, C]
                logits_list.append(output)
                predictions.append(output.argmax(dim=1))
            #print("predictions", predictions)

            if vote == 'majority':
                stacked = torch.stack(predictions)  # [5, 1]
                final = stacked.mode(dim=0).values.squeeze(0)
                final_logit = logits_list[0]# scalar
            else:  # maxlogit
                stacked_logits = torch.stack(logits_list)
                max_logits, _ = torch.max(stacked_logits, dim=0)# [5, 1, C]
                final = torch.argmax(max_logits)
                final_logit = max_logits

            final_preds.append(final)
            final_logits.append(final_logit.unsqueeze(0))

        return torch.stack(final_preds), torch.cat(final_logits, dim=0)

def predict_with_weighted_topk_experts_batch(selector, experts, videos, device, vote='maxlogit'):
    with torch.no_grad():
        B = videos.size(0)
        selector_output = selector(videos.to(device))
        selector_logits = selector_output if isinstance(selector_output, torch.Tensor) else selector_output[0]  # [B, num_experts]
        topk = 5  # you can parametrize this if needed
        topk_indices = torch.topk(selector_logits, k=topk, dim=1).indices  # [B, topk]

        final_preds = []
        final_logits = []

        selector_softmax = torch.softmax(selector_logits, dim=1)  # [B, num_experts]

        for i in range(B):
            video = videos[i].unsqueeze(0)  # [1, C, T, H, W]
            topk_experts = topk_indices[i].tolist()

            logits_list = []
            for k in topk_experts:
                logit = experts[k](video.to(device))[0]  # [1, num_classes_per_expert]
                weight = selector_softmax[i][k].item()
                weighted_logit = logit * weight
                logits_list.append(weighted_logit)

            stacked_logits = torch.stack(logits_list)  # [topk, 1, C_k]
            max_logits, _ = torch.max(stacked_logits, dim=0)  # [1, C_k]
            final = torch.argmax(max_logits)
            final_preds.append(final)
            final_logits.append(max_logits.unsqueeze(0))

        return torch.stack(final_preds), torch.cat(final_logits, dim=0)


@torch.no_grad()
def final_test(args, data_loader, selector, experts, device, file):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    selector.eval()
    for expert in experts:
        expert.eval()

    final_result = []
    labels = []
    Out = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]

        batch_size = samples.shape[0]
        samples = samples.to(device, non_blocking=True)
        target = torch.tensor(target).to(device, non_blocking=True)
        labels.extend(target.cpu().numpy())

        with torch.no_grad():
            output = []
            preds, final_logits = predict_with_topk_experts_batch(selector, experts, samples, device, vote=args.vote_strategy)
            Out.extend(preds.cpu().numpy())
            print("target:",target, "prediction", preds)

        
        for i in range(batch_size):
            expert_logits = final_logits.data[i].cpu().numpy().tolist()
            string = "{} {} {} {} {}\n".format(ids[i], \
                                              str(expert_logits[0]), \
                                              str(int(target[i].cpu().numpy())), \
                                              str(int(chunk_nb[i].cpu().numpy())), \
                                              str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)
            
       
            
        #print("final logits shape", final_logits.shape)
        acc1 = (preds == target).float().sum() * 100.0 / target.size(0)
        acc5= acc1
        #metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} '
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    print("Return statement")
    print("using expert_logits[0]")
    print("using weighted logits")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        



@torch.no_grad()
def final_test_individual(args, data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    labels=[]
    
    Out=[]
    
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        target = batch[1]
        
        #print(batch,"batch-----------")
        
        target = torch.tensor(target)
        labels.extend(target.cpu().numpy())
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        batch_size = samples.shape[0]
        samples = samples.to(device, non_blocking=True)
        #print(f" samoles: {samples.shape}")
        print(f" target: {target}")
        #print(f" ids: {ids}")
      
        target = target.to(device, non_blocking=True)
        #print(samples)
        # compute output
        with torch.cuda.amp.autocast():
        
            output,y,s_x,t_x = model(samples)
            y_array = y.cpu().numpy()
            s_xarray=s_x.cpu().numpy()
            t_xarray=t_x.cpu().numpy()
            
            print("type of y",type(y_array))
            if 'features' in locals():
                features = np.vstack((features, y_array))
            else:
                features = y_array
            if 'spatial_features' in locals():
                spatial_features=np.vstack((spatial_features,s_xarray))
            else:
                spatial_features=s_xarray
                
            if 'temp_features' in locals():
                temp_features=np.vstack((temp_features,t_xarray))
            else:
                temp_features=t_xarray
                
                

            #print(output.shape)
            #print("output", output[0])
            Out.extend(np.argmax(output.cpu().numpy(), axis=1))
            #
            
            loss = criterion(output, target)
           # print(output)
    
        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
   
    
    import pandas as pd
    

    def plot_final_features(features,category):
    
        tsne = TSNE(n_components=2, random_state=42)

        
        
        X_tsne = tsne.fit_transform(features)
        df =pd.DataFrame(X_tsne, columns=['x','y'])
        df['category']=category
        groups=df.groupby('category')
        
        fig, ax= plt.subplots()
        for name, points in groups:
            ax.scatter(points.x, points.y, label=name)
        
        
        ax.legend()
        plt.savefig(f'group-4.png')
    
    def plot_temporal_features(temp_features,category):
        tsne = TSNE(n_components=2, random_state=42)

        
        
        tx_tsne = tsne.fit_transform(temp_features)
        df =pd.DataFrame(tx_tsne, columns=['x','y'])
        df['category']=category
        groups=df.groupby('category')
        
        fig, ax= plt.subplots()
        for name, points in groups:
            ax.scatter(points.x, points.y, label=name)
            
        plt.title("Plot of temporal features")
            
        ax.legend()
        plt.show()
        
    def plot_spatial_features(spatial_features,category):
        tsne = TSNE(n_components=2, random_state=42)

        
        
        sx_tsne = tsne.fit_transform(spatial_features)
        df =pd.DataFrame(sx_tsne, columns=['x','y'])
        df['category']=category
        groups=df.groupby('category')
        
        fig, ax= plt.subplots()
        for name, points in groups:
            ax.scatter(points.x, points.y, label=name)
            
        ax.legend()
        plt.title("Plot of spatial features")
        plt.show()
    
    def plot_confusion_matrix(y_true,y_pred):
               
    
        from sklearn.metrics import confusion_matrix
        import numpy as np

        # Assuming true_labels and predicted_labels are your ground truth and predictions, respectively
        # Replace these lists with your actual true and predicted labels
        true_labels = y_true # List of true labels (length = 1000+)
        predicted_labels = y_pred # List of predicted labels (length = 1000+)

        # Number of classes (174 in this case)
        num_classes = 174

        # Calculate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))

        # Initialize an array to store recall values for each class
        recalls = []

        # Calculate recall for each class
        for i in range(num_classes):
            # True Positives for class i (cm[i, i])
            # False Negatives for class i (sum of column i, minus True Positives)
            true_positives = cm[i, i]
            false_negatives=cm[i, :].sum() - true_positives
            
            # Recall for class i = True Positives / (True Positives + False Negatives)
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
            recalls.append(recall)

        # Optionally, print recall for each class
        for i in range(num_classes):
            print(f'Recall for class {i}: {recalls[i]}')


    
    #plot_confusion_matrix(labels,Out)
        
        
    ##plot_spatial_features(spatial_features,labels)
    ##plot_temporal_features(temp_features,labels)
    #plot_final_features(features,labels)   
    ##plot_spatial_features(spatial_features,Out)
    ##plot_temporal_features(temp_features,Out)
    #plot_final_features(features,Out)  
        
    
    

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print("Return statement")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float64, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
