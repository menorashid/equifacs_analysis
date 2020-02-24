from helpers import util, visualize
from script_cumulative_separation import get_feats
import numpy as np
import os
import cooc

class Base_Selector():
    def __init__(self, inc, step_size, feature_type, pain):
        self.inc = inc
        self.step_size = step_size
        self.feature_type = feature_type 
        self.pain = pain

    def get_feats_by_type(self, feature_type):
        features_all = []
        aus_all = []
        labels_all = []
        for feature_type_curr in feature_type:
            features, labels, all_aus, _ =get_feats(self.inc, self.step_size, data_type = feature_type_curr)
            if len(feature_type)>1:
                features_all += [features]
                labels_all += [labels]
                aus_all += [all_aus]

        if len(feature_type)>1:
            for idx in range(len(features_all)):
                assert np.all(labels_all[0]==labels_all[idx])
                assert np.all(np.array(aus_all[0])==np.array(aus_all[idx]))
            features = np.concatenate(features_all, axis = 1)
            labels = labels_all[0]
            all_aus_new = []
            for f_type in feature_type:
                all_aus_new+=[au_curr+'_'+f_type for au_curr in all_aus]
            all_aus = all_aus_new

        return features, labels, all_aus

    def create_splits(self, features, labels, all_aus):

        idx_test_all = []
        classes_keep_all = []
        for test_label in np.unique(labels):
            # idx_test = labels == test_label
            idx_test_all.append(test_label)
            classes_keep_all.append(np.array(all_aus))
        return idx_test_all, classes_keep_all

    def select_and_split(self):
        features, labels, all_aus = self.get_feats_by_type(self.feature_type) 
        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus)
        
        idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus, classes_keep_all, idx_test, labels)

        return features, labels, all_aus, idx_test_all, bin_keep_aus

    def convert_vals_to_bins(self, all_aus_used, classes_keep_all, idx_test, labels):
        bin_keep_aus = []
        for classes_keep in classes_keep_all:
            bin_keep = np.in1d(all_aus_used, classes_keep)
            bin_keep_aus.append(bin_keep)

        idx_test_all = []
        for idx_test_curr in idx_test:
            idx_test_all.append(labels == idx_test_curr)

        return idx_test_all, bin_keep_aus


class Cooc_Selector(Base_Selector):
    def __init__(self, inc, step_size, feature_type, pain, thresh):
        Base_Selector.__init__(self,inc, step_size, feature_type, pain)
        self.thresh = thresh

    def create_splits(self, features, labels, all_aus):
        idx_test_all = []
        classes_keep_all = []
        for test_label in np.unique(labels):
            idx_test = labels == test_label
            idx_train = np.logical_not(idx_test)
            classes, diffs = cooc.find_best_clusters_custom(features[idx_train,:], labels[idx_train], all_aus, self.pain, plot_it = False)

            diffs = diffs - np.min(diffs)
            diffs = diffs/np.max(diffs)
            classes_keep = classes[diffs>self.thresh]

            idx_test_all.append(test_label)
            classes_keep_all.append(classes_keep)

            
        return idx_test_all, classes_keep_all

    def select_and_split(self):
        features, labels, all_aus = self.get_feats_by_type(['frequency'])

        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus)
        

        features, labels, all_aus = self.get_feats_by_type(self.feature_type)

        all_aus_used = np.array([au.split('_')[0] for au in all_aus])
        
        idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, idx_test, labels)
        # print np.array(all_aus)[bin_keep_aus[0]]

        return features, labels, all_aus, idx_test_all, bin_keep_aus

class Kunz_Selector(Base_Selector):
    def __init__(self, inc, step_size, feature_type, pain):
        Base_Selector.__init__(self,inc, step_size, feature_type, pain)
        self.thresh = 0.05


    def select(self, features, labels, all_aus):
        bin_pain = np.in1d(labels, self.pain)
        pain_totals = np.sum(features[bin_pain,:],axis = 0)
        percentages = pain_totals/float(np.sum(pain_totals))
        aus_keep_1st = percentages>self.thresh
        no_pain_totals = np.sum(features[np.logical_not(bin_pain),:],axis = 0)
        aus_keep_2nd = np.logical_and(aus_keep_1st, pain_totals>no_pain_totals)
        classes_keep = np.array(all_aus)[aus_keep_2nd]
        return classes_keep

    def create_splits(self, features, labels, all_aus):
        # labels, labels_true = labels
        idx_test_all = []
        classes_keep_all = []
        for test_label in np.unique(labels):
            idx_test = labels == test_label
            idx_train = np.logical_not(idx_test)
            classes_keep = self.select(features[idx_train,:], labels[idx_train], all_aus)
            idx_test_all.append(test_label)
            classes_keep_all.append(classes_keep)

            
        return idx_test_all, classes_keep_all

    def select_and_split(self):
        features, labels, all_aus, _ =get_feats(inc=30, step_size = 30, data_type = 'frequency')
        
        
        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus)
        features, labels, all_aus = self.get_feats_by_type(self.feature_type)


        all_aus_used = np.array([au.split('_')[0] for au in all_aus])
        
        idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, idx_test, labels)
        

        return features, labels, all_aus, idx_test_all, bin_keep_aus



