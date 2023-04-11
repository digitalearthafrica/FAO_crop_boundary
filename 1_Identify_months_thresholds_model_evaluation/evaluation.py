import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

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