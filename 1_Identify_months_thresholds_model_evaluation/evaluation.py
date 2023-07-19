import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from tqdm import tqdm
from mxnet import autograd

def Calculate_IoUs(instances_true, instance_predicted, plot=False):
    """
    compute IoUs and true fied sizes
    
    INPUTS:
    instances_true: labelled true field extent
    instance_predicted : predicted and labelled field extent instances
    plot: whether to plot results
    """
    
    # plot true and predicted fields
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(instances_true)
        ax[1].imshow(instance_predicted)
        plt.show()
    
    best_IoUs = [] # best IoU for each true field
    field_sizes = [] # field sizes
        
    # get unique values for all true fields
    field_values = np.unique(instances_true)
    # print('number of groundtruth fields',len(field_values))
    # loop through true fields
    for field_value in field_values:
        if field_value == 0:# skip background or noncrop (0)
            continue 
        # identify current true field
        this_field = instances_true == field_value
        # calculate current field size
        field_sizes.append(np.sum(this_field))
        
        # find label values that intersect with true field
        intersecting_fields = this_field * instance_predicted
        # find unique values of the intersecting labels
        intersect_values = np.unique(intersecting_fields)
        # find corresponding fields
        intersect_fields = np.isin(instance_predicted, intersect_values)
        # plot all intersecting fields on the prediction
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(this_field)
            ax[1].imshow(intersect_fields)
            plt.show()
        # compute IoUs for all intersecting fields
        field_IoUs = []
        for intersect_value in intersect_values: # skip noncrop (0)
            if intersect_value == 0:
                continue
            pred_field = instance_predicted == intersect_value
            union = this_field + pred_field > 0
            intersection = (this_field * pred_field) > 0
            IoU = np.sum(intersection) / np.sum(union)
            field_IoUs.append(IoU)
        # take maximum IoU - this is the IoU for this true field
        if len(field_IoUs) > 0:
            best_IoUs.append(np.max(field_IoUs))
        else:
            best_IoUs.append(0)
    
    return best_IoUs, field_sizes

def get_accuracy_scores(extent_true,boundary_true,extent_prob_predicted):
    '''function to calculate pixel-based accuracy scores: overall accuracy, f1 and mcc'''
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    # binarise predicted extent
    # NOTE: you may want to fine tune this 0.5 threshold to get a fair judgement on semantic segmentation results
    # e.g. when your predicted probabilities are overall higher or lower
    extent_predicted=np.ceil(extent_prob_predicted-0.5) 
    # define evaluation mask area, outside of which won't be evaluated
    # evaluation_mask=(extent_predicted==segmentation.clear_border(extent_predicted,bgval=0))
    # evaluation_mask=(instance_predicted==segmentation.clear_border(instance_predicted,buffer_size=2,bgval=0))
    evaluation_mask=(extent_true+boundary_true)>=1

    evaluation_mask_reshaped=evaluation_mask.reshape((evaluation_mask.shape[0]*evaluation_mask.shape[1],1))
    # find pixel indices of mask
    nonmasked_idx=evaluation_mask_reshaped==1

    # apply the evaluation mask to ground truth extent
    extent_true_reshaped=extent_true.reshape((extent_true.shape[0]*extent_true.shape[1], 1))
    extent_true_masked=extent_true_reshaped[nonmasked_idx]

    # apply the evaluation mask to predicted extent
    extent_predicted_reshaped=extent_predicted.reshape((extent_predicted.shape[0]*extent_predicted.shape[1], 1))
    extent_predicted_masked=extent_predicted_reshaped[nonmasked_idx]

    # apply the evaluation mask to predicted extent probability
    extent_prob_predicted_reshaped=extent_prob_predicted.reshape((extent_prob_predicted.shape[0]*extent_prob_predicted.shape[1], 1))
    extent_prob_predicted_masked=extent_prob_predicted_reshaped[nonmasked_idx]

    # calculate accuracy
    accuracy.update(mx.nd.array(extent_true_masked), mx.nd.array(extent_predicted_masked)) # remember to convert arrays to mxnet ndarray
    
    # f1 score
    # probabilities = mx.nd.stack(extent_prob_predicted_masked, 1 - extent_prob_predicted_masked, axis=1)
    probabilities = np.stack((1 - extent_prob_predicted_masked,extent_prob_predicted_masked), axis=1)
    f1.update(mx.nd.array(extent_true_masked), mx.nd.array(probabilities))

    # MCC metric
    mcc.update(mx.nd.array(extent_true_masked), mx.nd.array(probabilities))
    return accuracy,f1,mcc

def dice_coef(x, y):
    '''function to calculate dice coefficient / F1 score'''
    if type(x).__module__ == 'numpy':
        intersection = np.logical_and(x, y)
        return 2. * np.sum(intersection) / (np.sum(x) + np.sum(y))
    else:
        intersection = mx.ndarray.op.broadcast_logical_and(x, y)
        return 2. * mx.nd.sum(intersection) / (mx.nd.sum(x) + mx.nd.sum(y))

def train_model_per_epoch(train_dataloader, model, ftnmt_loss, trainer, epoch, args):
    """
    function to train the model and calculate training scores for one epoch
    
    INPUTS:
    val_dataloader: DataLoader to iterate through the dataset in mini-batches
    model: the nn model
    mtsk_loss: multitask tanimoto loss function: takes in prediction and true label
    trainer: 
    epoch: epoch number
    args: other arguments including batch size, cpu/gpu and visulisation
    """
    # initialize metrics
    cumulative_loss = 0
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    dice = mx.metric.CustomMetric(feval=dice_coef, name="Dice")
    if args['ctx_name'] == 'cpu':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args['gpu'])
    
    # training set
    for batch_i, (img, extent, boundary, distance, mask) in enumerate(
        tqdm(train_dataloader, desc='Training epoch {}'.format(epoch))):
        
        with autograd.record():
            # make a copy if the variable currently lives in the wrong context
            img = img.as_in_context(ctx) 
            extent = extent.as_in_context(ctx)
            boundary = boundary.as_in_context(ctx)
            distance = distance.as_in_context(ctx)
            mask = mask.as_in_context(ctx)
            
            # predicted outputs: field extent probability, field boundary and distance to boundary
            logits, bound, dist = model(img)
        
            # Fractal Tanimoto loss (basically Jaccard distance?)
            loss_extent = mx.nd.sum(ftnmt_loss(logits*mask, extent*mask))
            loss_boundary = mx.nd.sum(ftnmt_loss(bound*mask, boundary*mask))
            loss_distance = mx.nd.sum(ftnmt_loss(dist*mask, distance*mask))
            loss = 0.33 * (loss_extent + loss_boundary + loss_distance)
            
#             # Multi-task fractal Tanimoto loss: didn't work
#             labels = mx.nd.concat(*[extent,boundary,distance],dim=1)
#             predictions=mx.nd.concat(*[logits,bound,dist],dim=1)
#             loss=mtsk_loss(predictions,labels)

        # compute the gradients w.r.t. the loss function 
        loss.backward()

        # Makes one step of parameter update
        trainer.step(args['batch_size'])
        
        # update cummulative loss 
        cumulative_loss += mx.nd.sum(loss).asscalar()
        
        # mask out unlabeled pixels
        logits_reshaped = logits.reshape((logits.shape[0], -1))
        extent_reshaped = extent.reshape((extent.shape[0], -1))
        mask_reshaped = mask.reshape((mask.shape[0], -1))

        nonmask_idx = mx.np.nonzero(mask_reshaped.as_np_ndarray())
        nonmask_idx = mx.np.stack(nonmask_idx).as_nd_ndarray().as_in_context(ctx)
        logits_masked = mx.nd.gather_nd(logits_reshaped, nonmask_idx)
        extent_masked = mx.nd.gather_nd(extent_reshaped, nonmask_idx)

        # update metrics based on every batch
        # update accuracy
        extent_predicted_classes = mx.nd.ceil(logits_masked - 0.5)
        accuracy.update(extent_masked, extent_predicted_classes)

        # f1 score
        probabilities = mx.nd.stack(1 - logits_masked, logits_masked, axis=1)
        f1.update(extent_masked, probabilities)

        # MCC metric
        mcc.update(extent_masked, probabilities)

        # Dice score
        dice.update(extent_masked, extent_predicted_classes)
        
    return cumulative_loss, accuracy, f1, mcc, dice

def evaluate_model_per_epoch(val_dataloader, model, ftnmt_loss, epoch, args):
    """
    function to run model instance and calculate evaluation scores for one epoch
    
    INPUTS:
    val_dataloader: DataLoader to iterate through the dataset in mini-batches
    model: the nn model
    mtsk_loss: tanimoto loss function: takes in prediction and true label
    epoch: epoch number
    args: other arguments including batch size, cpu/gpu and visulisation
    """
    
    # initialize metrics
    cumulative_loss = 0
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    dice = mx.metric.CustomMetric(feval=dice_coef, name="Dice")
    
    if args['ctx_name'] == 'cpu':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args['gpu'])
    
    # validation set
    for batch_i, (img, extent, boundary, distance, mask) in enumerate(
        tqdm(val_dataloader, desc='Validation epoch {}'.format(epoch))):

        # make a copy if the variable currently lives in the wrong context
        img = img.as_in_context(ctx)
        extent = extent.as_in_context(ctx)
        boundary = boundary.as_in_context(ctx)
        distance = distance.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        
        # predicted outputs: field extent probability, field boundary probability and distance to boundary
        logits, bound, dist = model(img)
        
        # Fractal Tanimoto loss (basically Jaccard distance?)
        loss_extent = mx.nd.sum(ftnmt_loss(logits*mask,extent*mask))
        loss_boundary = mx.nd.sum(ftnmt_loss(bound*mask,boundary*mask))
        loss_distance = mx.nd.sum(ftnmt_loss(dist*mask,distance*mask))
        loss = 0.33 * (loss_extent + loss_boundary + loss_distance)
            
        # update cummulative loss
        cumulative_loss += mx.nd.sum(loss).asscalar()

        # mask out unlabeled pixels
        logits_reshaped = logits.reshape((logits.shape[0], -1))
        extent_reshaped = extent.reshape((extent.shape[0], -1))
        mask_reshaped = mask.reshape((mask.shape[0], -1))

        nonmask_idx = mx.np.nonzero(mask_reshaped.as_np_ndarray())
        nonmask_idx = mx.np.stack(nonmask_idx).as_nd_ndarray().as_in_context(ctx)
        logits_masked = mx.nd.gather_nd(logits_reshaped, nonmask_idx)
        extent_masked = mx.nd.gather_nd(extent_reshaped, nonmask_idx)
        
        # update metrics based on every batch
        # update accuracy
        extent_predicted_classes = mx.nd.ceil(logits_masked - 0.5)
        accuracy.update(extent_masked, extent_predicted_classes)

        # f1 score
        probabilities = mx.nd.stack(1 - logits_masked, logits_masked, axis=1)
        f1.update(extent_masked, probabilities)

        # MCC metric
        mcc.update(extent_masked, probabilities)

        # Dice score
        dice.update(extent_masked, extent_predicted_classes)
        
    return cumulative_loss, accuracy, f1, mcc, dice