from helpers import util, visualize
from script_cumulative_separation import get_feats
import numpy as np
import os
import math
from clustering import fit_lda, get_scaler
import cooc
import itertools
import sklearn.metrics
import feature_selectors
import copy
from doing_ml import select_features
import scipy.stats
import statsmodels.api as sm
from linear_regressor import LinearRegression

def script_p_value(out_dir_meta, ows, feature_types, feature_selection, selection_params, norm, alpha, reverse = False):

    pain = np.array([1,2,4,5,11,12])
    selection_params = copy.deepcopy(selection_params)
    selection_params['pain']=pain

    iterator = itertools.product(ows,feature_types)
    for ows_curr, feature_type in iterator:
        inc,step_size = ows_curr

        dir_str = '_'.join([str(val) for val in ['inc',inc,'step',step_size]])
        feature_type_str = '_'.join(feature_type)
        out_dir = os.path.join(out_dir_meta,dir_str,feature_type_str)
        print out_dir
        # util.mkdir(out_dir)
        
        selection_params['inc']=inc
        selection_params['step_size']=step_size
        selection_params['feature_type']=feature_type
        train_package, test_package = select_features(selection_type = feature_selection, selection_params = selection_params, clinical = True)
        # ['all_aus', 'labels', 'features', 'bin_keep_aus']

        features = train_package['features']
        all_aus = train_package['all_aus']
        labels = train_package ['labels']
        class_pain = np.in1d(labels, pain).astype(int)
        if reverse:
            print class_pain
            class_pain = np.logical_not(class_pain).astype(int)
            print class_pain
        scaler = get_scaler(norm)
        features = scaler.fit_transform(features)
        model = sm.OLS(class_pain,features)
        res = model.fit()
        # .f_pvalue
        print res.pvalues
        # print p_value

        p_values = []
        for idx in range(features.shape[1]):
            x_curr = features[:,idx]
            # x_curr_n = sm.add_constant(x_curr)
            model = sm.OLS(class_pain,x_curr)
            res = model.fit()
            print res.params
            print res.pvalues
            
            p_value = res.f_pvalue
            print p_value
            raw_input()
            # # print results
            # # p_values = fii.summary2()
            # # # .tables[1]['P>|t|']
            # # print p_values
            # # raw_input()
            # print p_value
            # slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(x_curr,class_pain)
            # lg = LinearRegression(normalize = False, fit_intercept = True)
            # lg.fit(x_curr[:,np.newaxis], class_pain[:,np.newaxis])
            # print lg.p
            # print p_value
            p_values.append(p_value)

        # raw_input()

        corrected = []
        p_values = np.array(p_values)
        idx_sort = np.argsort(p_values)
        ranks = np.array(range(len(p_values)))
        p_values_sorted = p_values[idx_sort]
        all_aus_sorted = np.array(all_aus)[idx_sort]
        # print ranks, features.shape[0], alpha
        bh = (ranks+1)/float(features.shape[0])*alpha

        for idx in range(p_values.size):
            if bh[idx]>=0.06:
                break
            print all_aus_sorted[idx], p_values_sorted[idx], bh[idx]

        break

def main():

    print 'hello'
    out_dir_meta = '../experiments/ml_on_aus_clinical'
    util.mkdir(out_dir_meta)
    eval_methods = ['raw','majority','atleast_1']

    ows = [[2,1],[5,2.5],[10,5],[15,7.5],[20,10],[30,30]]
    # ows = [[10,5]]
    # feature_types = [['frequency'],['duration']]
    # ,['frequency','duration']]
    
    # ows = [[30,30]]
    feature_types = [['frequency']]
    # ows = [[5,2.5]]

    feature_selection = None
    selection_params = {}
    norm = 'mean_std'
    alpha = 0.25
    reverse = True
    # feature_selection = 'cooc'
    # selection_params = dict(thresh=0.5)
    
    # feature_selection = 'kunz'
    # selection_params = {}

    
    # feature_selection = 'collapse'
    # selection_params = dict(select_mode=None,flicker=1,brow=0)

    # selection_params = dict(select_mode='cooc',flicker=0,brow=0, thresh = 0.7)
    # selection_params = dict(select_mode='kunz',flicker=0,brow=0, thresh = 0.05)
    
    # test_clinical(out_dir_meta, ows, feature_types, feature_selection,selection_params, eval_methods = eval_methods)
    script_p_value(out_dir_meta, ows, feature_types, feature_selection, selection_params, norm, alpha, reverse = reverse)



if __name__=='__main__':
    main()
