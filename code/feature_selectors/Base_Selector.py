from script_cumulative_separation import get_feats
import numpy as np
import cooc
from read_in_data import read_clinical_pain
class Base_Selector():
    def __init__(self, inc, step_size, feature_type, pain, type_dataset):
        self.inc = inc
        self.step_size = step_size
        self.feature_type = feature_type 
        self.pain = pain
        # print type_dataset
        self.type_dataset = type_dataset

    def get_feats_by_type(self, feature_type, clinical = False, flicker = 0):
        features_all = []
        aus_all = []
        labels_all = []
        for feature_type_curr in feature_type:
            features, labels, all_aus, pain =get_feats(self.inc, self.step_size, data_type = feature_type_curr, clinical = clinical,type_dataset = self.type_dataset, flicker = flicker)
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

        # bin_pain = np.in1d(labels, pain).astype(int)
        return features, labels, all_aus, pain

    def create_splits( self,features, labels, all_aus, pain, no_test = False):
        
        # labels, labels_true = labels
        # print 'base',pain, self.type_dataset
        # raw_input()

        idx_test_all = []
        classes_keep_all = []
        for test_label in np.unique(labels):
            
            # idx_test = labels == test_label
            idx_test_all.append(test_label)
            classes_keep_all.append(np.array(all_aus))
            print (all_aus.shape)
            
            if no_test:
                idx_test_all = [None]
                break

        return idx_test_all, classes_keep_all

    def select_and_split(self):
        features, labels, all_aus = self.get_feats_by_type(self.feature_type) 
        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus)
        idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus, classes_keep_all, idx_test, labels)
        # print all_aus
        # print classes_keep_all
        # print bin_keep_aus
        # raw_input()
        return features, labels, all_aus, idx_test_all, bin_keep_aus

    def select_exp_clinical(self):
        features, labels, all_aus = self.get_feats_by_type(self.feature_type) 
        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, no_test = True)

        features_clinical, labels_clinical, all_aus_clinical = self.get_feats_by_type(self.feature_type, clinical = True)

        classes_keep_all = [np.intersect1d(classes_keep_all[0], all_aus_clinical)]

        _, bin_keep_aus = self.convert_vals_to_bins( all_aus, classes_keep_all, [], [])
        _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_clinical, classes_keep_all, [], [])
        
        assert np.all(np.array(all_aus_clinical)[bin_keep_aus_clinical[0]] ==np.array(all_aus)[bin_keep_aus[0]])

        pain_dict = read_clinical_pain()
        pain_arr = np.zeros((labels_clinical.shape[0], 3))

        for idx_label, label in enumerate(labels_clinical):
            pain_arr[idx_label,:] = pain_dict[label]

        train_package = {}
        train_package['features'] = features
        train_package['labels'] = labels
        train_package['all_aus'] = np.array(all_aus)
        train_package['bin_keep_aus'] = bin_keep_aus[0]

        test_package = {}
        test_package['features'] = features_clinical
        test_package['labels'] = labels_clinical
        test_package['all_aus'] = np.array(all_aus_clinical)
        test_package['bin_keep_aus'] = bin_keep_aus_clinical[0]
        test_package['pain'] = pain_arr
        
        # for k in train_package.keys():
        #     print k, train_package[k].shape

        # for k in test_package.keys():
        #     print k, test_package[k].shape
        # raw_input()
        return train_package, test_package

    def convert_vals_to_bins(self, all_aus_used, classes_keep_all, idx_test, labels):
        bin_keep_aus = []
        for classes_keep in classes_keep_all:
            bin_keep = np.in1d(all_aus_used, classes_keep)
            bin_keep_aus.append(bin_keep)

        idx_test_all = []
        for idx_test_curr in idx_test:
            idx_test_all.append(labels == idx_test_curr)

        return idx_test_all, bin_keep_aus

