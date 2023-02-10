import os
import higra as hg
import numpy as np
from osgeo import gdal

def InstSegm(extent, boundary, t_ext=0.4, t_bound=0.2):
    """
    function to do instance segmentation from predicted extent and boundary probabilites
    
    INPUTS:
    extent : extent probability prediction 
    boundary : boundary probability prediction
    t_ext : threshold for extent
    t_bound : threshold for boundary
    OUTPUT:
    instances segmentation (noncrop:-1)
    """

    # Threshold extent mask
    ext_binary = np.uint8(extent >= t_ext)

    # Artificially create strong boundaries for pixels with non-field labels
    input_hws = np.copy(boundary)
    input_hws[ext_binary == 0] = 1

    # Create the directed graph
    size = input_hws.shape[:2]
    graph = hg.get_8_adjacency_graph(size)
    edge_weights = hg.weight_graph(
        graph,
        input_hws,
        hg.WeightFunction.mean
    )
    # Watershed hierarchy by dynamics
    tree, altitudes = hg.watershed_hierarchy_by_dynamics(
        graph,
        edge_weights
    )
    
    # Get individual fields by cutting the graph using altitude
    instances = hg.labelisation_horizontal_cut_from_threshold(
        tree,
        altitudes,
        threshold=t_bound)
    
    # assign non-field labels back to background value (-1)
    instances[ext_binary == 0] = -1

    return instances