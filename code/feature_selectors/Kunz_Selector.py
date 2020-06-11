from .Base_Selector import *

class Kunz_Selector(Base_Selector):
    def __init__(self,inc=0, step_size=0, feature_type=None, pain=None, type_dataset = 'isch', flicker = 0):
        # print type_dataset
        Base_Selector.__init__(self,inc, step_size, feature_type, pain, type_dataset = type_dataset)
        self.thresh = 0.05
        self.flicker = flicker


    # def select(self, features, labels, all_aus):
    #     bin_pain = np.in1d(labels, self.pain)
    #     pain_totals = np.sum(features[bin_pain,:],axis = 0)
    #     percentages = pain_totals/float(np.sum(pain_totals))
    #     aus_keep_1st = percentages>self.thresh
    #     no_pain_totals = np.sum(features[np.logical_not(bin_pain),:],axis = 0)
    #     aus_keep_2nd = np.logical_and(aus_keep_1st, pain_totals>no_pain_totals)
    #     classes_keep = np.array(all_aus)[aus_keep_2nd]
    #     return classes_keep

    def create_splits(self, features, labels, all_aus, pain=None , thresh=None, no_test = False ):
        # labels, labels_true = labels
        # print 'kunz',pain, self.type_dataset
        # raw_input()

        debug = True
        
        if pain is None:
            pain = self.pain
        if thresh is None:
            thresh = self.thresh

        idx_test_all = []
        classes_keep_all = []


        if no_test:
            test_labels = [None]
        else:
            test_labels = np.unique(labels)


        features_train = features
        bin_pain = np.in1d(labels, pain)
        num_pain = np.sum(bin_pain)
        num_no_pain = np.sum(np.logical_not(bin_pain))

        print labels, bin_pain,num_pain, num_no_pain
        # raw_input()

        pain_totals = np.sum(features_train[bin_pain,:],axis = 0)/float(num_pain)
        percentages = pain_totals/float(np.sum(pain_totals))
        
        aus_keep_1st = percentages>thresh
        no_pain_totals = np.sum(features_train[np.logical_not(bin_pain),:],axis = 0)/float(num_no_pain)
        percentages_no_pain = no_pain_totals/float(np.sum(no_pain_totals))

        aus_keep_2nd = np.logical_and(aus_keep_1st, pain_totals>no_pain_totals)
        classes_keep = np.array(all_aus)[aus_keep_2nd]
        # print classes_keep

        
        if debug:
            # manual_bin_keep = np.in1d(all_aus, ['au101','au5'])
            # manual_bin_keep = np.in1d(all_aus, ['ad81'])
            # aus_keep_1st = manual_bin_keep
            # manual_bin_keep = np.in1d(all_aus,['ad1','ad38','au101','au17','au47','ead104'])
            # ,'au10','au18','ad81'])
                # ['ad38','au17','au47','ead104','au101','ad1'])
                # ['ad38','ad81','au17','au18','au47','auh13','ead104'])
                
            # classes_keep = np.array(all_aus)[manual_bin_keep]
            # aus >thresh
            str_p = '\t'.join([val.upper() for val in np.array(all_aus)[aus_keep_1st]])
            print str_p
            
            # aus >thresh percentages in pain
            str_p = '\t'.join(['%.2f'%(val*100)+"%" for val in percentages[aus_keep_1st]])
            print str_p

            # aus >thresh percentages in no pain
            str_p = '\t'.join(['%.2f'%(val*100)+"%" for val in percentages_no_pain[aus_keep_1st]])
            print str_p
            

            counts = [pain_totals[aus_keep_1st],no_pain_totals[aus_keep_1st]]
            
            # percentage difference
            str_p = '\t'.join(['%.2f'%(val*100)+"%" for val in (counts[0]-counts[1])/((counts[0]+counts[1])/2.)])
            print str_p
            
            # # raw au counts, and total occurrences for pain and no pain
            # str_p = '\t'.join(['%d'%(val*num_pain) for val in pain_totals[aus_keep_1st]])
            # print str_p, num_pain, np.sum(pain_totals*num_pain)
            # str_p = '\t'.join(['%d'%(val*num_no_pain) for val in no_pain_totals[aus_keep_1st]])
            # print str_p,num_no_pain, np.sum(no_pain_totals*num_no_pain)
            print classes_keep
            raw_input()

        for test_label in test_labels:
            if test_label is None:
                idx_test = np.zeros(labels.shape)>1
            else:
                idx_test = labels == test_label
            # idx_train = np.logical_not(idx_test)

            # features_train = features[idx_train,:]
            # bin_pain = np.in1d(labels[idx_train], pain)
            # pain_totals = np.sum(features_train[bin_pain,:],axis = 0)
            # percentages = pain_totals/float(np.sum(pain_totals))
            # aus_keep_1st = percentages>thresh
            # no_pain_totals = np.sum(features_train[np.logical_not(bin_pain),:],axis = 0)
            # aus_keep_2nd = np.logical_and(aus_keep_1st, pain_totals>no_pain_totals)
            # classes_keep = np.array(all_aus)[aus_keep_2nd]



            idx_test_all.append(test_label)
            classes_keep_all.append(classes_keep)

            
        return idx_test_all, classes_keep_all

    def select_and_split(self):
        features, labels, all_aus, pain =get_feats(inc=30, step_size = 30, data_type = 'frequency', type_dataset = self.type_dataset, flicker = self.flicker)

        # print features.shape

        
        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, pain = pain)
        features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type, flicker = self.flicker)


        all_aus_used = np.array([au.split('_')[0] for au in all_aus])
        
        idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, idx_test, labels)
        
        class_pain = np.in1d(labels, pain).astype(int)
        return features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus

    def select_exp_clinical(self):
        features, labels, all_aus, pain =get_feats(inc=30, step_size = 30, data_type = 'frequency', type_dataset = self.type_dataset)
        
        
        idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, no_test = True, pain = pain)

        features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type, flicker = self.flicker)

        features_clinical, labels_clinical, all_aus_clinical, _ = self.get_feats_by_type(self.feature_type, clinical = True, flicker = self.flicker)

        
        all_aus_used = np.array([au.split('_')[0] for au in all_aus])
        # print all_aus_used
        # print classes_keep_all
        _, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])

        all_aus_used = np.array([au.split('_')[0] for au in all_aus_clinical])
        # print all_aus_used
        # print classes_keep_all
        _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])
        # raw_input()
        assert np.all(np.array(all_aus_clinical)[bin_keep_aus_clinical[0]] ==np.array(all_aus)[bin_keep_aus[0]])

        # pain_labels_clinical, pain_levels_clinical 
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

        return train_package, test_package

