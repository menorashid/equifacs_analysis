from helpers import util, visualize
from cooc import find_best_clusters_custom
from script_cumulative_separation import get_feats, plot_prob_classification
import numpy as np
import os
import math

def main():
    inc = 5
    step_size = 2.5
    separator = 50
    flicker = 1
    blink = 1
    pain = np.array([1,2,4,5,11,12])
    feat_keep = ['au','ead','ad']
    
    out_dir = '../experiments/collapsing_aus'
    util.mkdir(out_dir)

    features, labels, all_aus, file_str =get_feats(inc, step_size,flicker = flicker,blink= blink)
    features[features>0]=1
    print all_aus
    
    collapsed_aus = [['au145', 'au143', 'au47'],
                    ['au5', 'ad1'],
                    ['au26', 'au25', 'ad160', 'au27'],
                    ['au17', 'au24', 'au16'],
                    ['au10', 'au12', 'au113'],
                    ['au18', 'au122'],
                    ['auh13', 'ad38', 'ad133'],
                    ['ad84', 'ad85', 'ad51', 'ad52', 'ad53', 'ad54', 'ad55', 'ad56']]

    collapsed_labels = ['blink','eyewide','openmouth','mouthtension','nasolabial','upperlip','nostrilwide','headmove']

    rest_labels = []
    for val in collapsed_aus:
        rest_labels = rest_labels+val
    rest_labels = list(set(rest_labels))
    rest_labels = np.logical_not(np.in1d(all_aus, rest_labels))
    # rest_labels = list(all_aus[rest_labels])
    feat_keep = collapsed_labels+['ead','ad','au','auh']

    # collapsed_labels+=list(all_aus[rest_labels])

    new_features = np.zeros((features.shape[0],len(collapsed_labels)))
    
    for idx_cau, (cau, cau_key) in enumerate(zip(collapsed_labels, collapsed_aus)):
        bin_keep = np.in1d(all_aus, cau_key)
        new_features[:,idx_cau] = np.sum(features[:,bin_keep], axis = 1)>0

    rest_features = features[:,rest_labels]
    rest_aus = np.array(all_aus)[rest_labels]
    new_features= np.concatenate([new_features, rest_features],axis =1)
    print new_features.shape
    caus = collapsed_labels+list(rest_aus)
    print len(collapsed_labels)

    find_best_clusters_custom(new_features, labels, caus, pain,feat_keep, out_dir, inc, step_size, plot_it = True)
    visualize.writeHTMLForFolder(out_dir, height=450, width=500) 
    core_aus = ['nostrilwide','mouthtension','ead104','au101_ead101_ead104','nasolabial']
    out_dir_class = os.path.join(out_dir,'classification_prob')
    util.mkdir(out_dir_class)
    plot_prob_classification(new_features, labels, caus, core_aus, out_dir_class)

if __name__=='__main__':
    main()