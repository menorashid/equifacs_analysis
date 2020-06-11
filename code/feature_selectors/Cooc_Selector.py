from .Base_Selector import *

class Cooc_Selector(Base_Selector):
    def __init__(self, inc=0, step_size=0, feature_type=None, pain=None, thresh=None, type_dataset = 'isch', flicker = 0):
        # print type_dataset
        Base_Selector.__init__(self,inc, step_size, feature_type, pain,type_dataset = type_dataset)
        self.thresh = thresh
        self.flicker = flicker

    def create_splits( self, features, labels, all_aus, pain=None, thresh=None, no_test = False):
        
        # print 'in cooc splits',pain, self.type_dataset
        # raw_input()
        if pain is None:
            pain = self.pain
        if thresh is None:
            thresh = self.thresh

        # print thresh
        idx_test_all = []
        classes_keep_all = []
        if no_test:
            test_labels = [None]
        else:
            test_labels = np.unique(labels)

        # print 'hello. we are in the eye of the tiger'
        # out_dir = '../experiments/cooc_simple'
        
        # classes, idx_keep = cooc.find_best_clusters_clique(features, labels, all_aus, pain, plot_it = False)
        # # print type(classes), type(idx_keep)
        # classes_keep = np.array(classes)[idx_keep]
        # print self.step_size, self.inc
        # print classes_keep
        # raw_input()
        
        
        # START OLD METHOD
        classes, diffs, percents = cooc.find_best_clusters_custom(features, labels, all_aus, pain, plot_it = False)
        # print diffs
        # min_diff = np.min(diffs[diffs>0])
        # print min_diff
        # raw_input()
        diffs = diffs - np.min(diffs)
        diffs = diffs/np.max(diffs)
        # thresh = float(np.max(diffs)*self.thresh)
        # print classes[diffs>thresh]
        # print diffs[diffs>thresh]

        # diffs = diffs/np.sum(diffs)
        # diffs = np.cumsum(diffs[::-1])[::-1]
        # print diffs

        if type(thresh)==float:
            classes_keep = classes[diffs>thresh]
            percents = [percent[diffs>thresh] for percent in percents]
        else:
            idx = np.argsort(diffs)[::-1]
            classes_keep = classes[idx[:thresh]]
        # print classes_keep
        # for idx,class_keep in enumerate(classes_keep):
        #     print classes_keep[idx],percents[0][idx],percents[1][idx],(percents[0]-percents[1])[idx]
        # print classes_keep
        # raw_input()
        # END OLD METHOD
        
        for test_label in test_labels:
            if test_label is None:
                idx_test = np.zeros(labels.shape)>1
            else:
                idx_test = labels == test_label
            idx_train = np.logical_not(idx_test)
            # classes, diffs = cooc.find_best_clusters_custom(features[idx_train,:], labels[idx_train], all_aus, pain, plot_it = False)

            # diffs = diffs - np.min(diffs)
            # diffs = diffs/np.max(diffs)
            # if type(thresh)==float:
            #     classes_keep = classes[diffs>thresh]
            # else:
            #     idx = np.argsort(diffs)[::-1]
            #     classes_keep = classes[idx[:thresh]]

            
            idx_test_all.append(test_label)
            classes_keep_all.append(classes_keep)
            # print classes_keep_all

        # print ('classes_keep',classes_keep)
        return idx_test_all, classes_keep_all

    def select_and_split(self):
        features, labels, all_aus,pain = self.get_feats_by_type(['frequency'], flicker = self.flicker)

        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, pain = pain)
        

        features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type, flicker = self.flicker)

        all_aus_used = np.array([au.split('_')[0] for au in all_aus ])
        idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, idx_test, labels)
        # print np.array(all_aus)[bin_keep_aus[0]]
        # print classes_keep_all
        class_pain = np.in1d(labels, pain).astype(int)
        return features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus

    def select_exp_clinical(self):
        features, labels, all_aus, pain = self.get_feats_by_type(['frequency'], flicker = self.flicker)

        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, pain = pain, no_test = True)
        features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type, flicker = self.flicker)
        features_clinical, labels_clinical, all_aus_clinical,_ = self.get_feats_by_type(self.feature_type, clinical = True, flicker = self.flicker)

        all_aus_used = np.array([au.split('_')[0] for au in all_aus])
        all_aus_used_clinical = np.array([au.split('_')[0] for au in all_aus_clinical])
        classes_keep_all = [np.intersect1d(classes_keep_all[0], all_aus_used_clinical)]

        _, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])
        _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_used_clinical, classes_keep_all, [], [])
        
        # print 'bin_keep_aus[0]', np.sum(bin_keep_aus[0]), bin_keep_aus[0].shape
        # print 'bin_keep_aus_clinical[0]', np.sum(bin_keep_aus_clinical[0]), bin_keep_aus_clinical[0].shape

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
        train_package['pain'] = pain

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
        # print classes_keep_all

        return train_package, test_package
        
