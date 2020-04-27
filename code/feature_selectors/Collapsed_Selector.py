from .Base_Selector import *
from .Cooc_Selector import *
from .Kunz_Selector import *

class Collapsed_Selector(Base_Selector):
    def __init__(self, inc, step_size, feature_type, pain= None, flicker=0, brow=0, thresh = None, select_mode = None, type_dataset = 'isch', keep_rest = False):
        Base_Selector.__init__(self,inc, step_size, feature_type, pain, type_dataset = type_dataset)

        self.flicker = flicker
        self.brow = brow
        self.select_mode = select_mode
        self.thresh = thresh
        self.keep_rest = keep_rest
        # if self.select_mode is None:
        #     self.thresh = None    
        # if self.select_mode=='cooc':
        #     self.thresh = float(self.select_mode.split('_')[1])
        # elif self.select_mode=='kunz':
        #     self.thresh = 0.05
        # else:
        #     raise ValueError('select_mode '+str(self.select_mode)+' is not valid')

        self.collapsed_aus = [['au145', 'au143', 'au47'],
                    ['au5', 'ad1'],
                    ['au26', 'au25', 'au27'],
                    ['au17', 'au24', 'au16'],
                    ['au10', 'au12', 'au113'],
                    ['au18', 'au122'],
                    ['auh13', 'ad38', 'ad133'],
                    ['ad84', 'ad85', 'ad51', 'ad52', 'ad53', 'ad54', 'ad55', 'ad56'],
                    ['ead103','ead104'],
                    ['au101']]
                    # 'ad160', 
        self.collapsed_labels = ['blink','eyewide','openmouth','mouthtension','nasolabial','upperlip','nostrilwide','headmove','earback','eyetension']

    def get_feats_by_type(self, feature_type, inc=None, step_size=None, clinical = False):
        
        if inc is None:
            inc = self.inc
        
        if step_size is None:
            step_size = self.step_size

        # print inc, step_size, clinical

        features_all = []
        aus_all = []
        labels_all = []
        for feature_type_curr in feature_type:
            # print feature_type_curr
            features, labels, all_aus, pain =get_feats(inc, step_size,flicker = self.flicker,blink= self.brow, data_type = feature_type_curr, clinical = clinical, type_dataset = self.type_dataset)
            # print all_aus
            # raw_input()
            # print features.shape
            # raw_input()
            if self.keep_rest:
                rest_labels = []
                for val in self.collapsed_aus:
                    rest_labels = rest_labels+val
                rest_labels = list(set(rest_labels))
                rest_labels = np.logical_not(np.in1d(all_aus, rest_labels))
            # rest_labels = list(all_aus[rest_labels])
            # feat_keep = self.collapsed_labels+['ead', 'au101']
            # ,'ad','au']

            new_features = np.zeros((features.shape[0],len(self.collapsed_labels)))
            
            for idx_cau, (cau, cau_key) in enumerate(zip(self.collapsed_labels, self.collapsed_aus)):

                bin_keep = np.in1d(all_aus, cau_key)
                # print np.array(all_aus)[bin_keep], cau_key
                # raw_input()
                if feature_type_curr=='duration' and (np.sum(bin_keep)>0):
                    new_features[:,idx_cau] = np.max(features[:,bin_keep], axis = 1)
                else:
                    new_features[:,idx_cau] = np.sum(features[:,bin_keep], axis = 1)

            caus = self.collapsed_labels

            if self.keep_rest:
                rest_features = features[:,rest_labels]
                rest_aus = np.array(all_aus)[rest_labels]
                new_features= np.concatenate([new_features, rest_features],axis =1)
                caus = caus+list(rest_aus)


            features = new_features
            # print features.shape
            # print caus
            # print features[0]
            # raw_input()
            all_aus = caus 

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

        return features, labels, all_aus, pain


    def select_and_split(self):
        if self.select_mode is None:
            # print self.feature_type
            features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type) 
            idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, pain = pain)
            idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus, classes_keep_all, idx_test, labels)
        elif self.select_mode=='cooc':
            features, labels, all_aus, pain = self.get_feats_by_type(['frequency'])
            idx_test, classes_keep_all = Cooc_Selector(type_dataset = self.type_dataset).create_splits(features, labels, all_aus, pain = pain, thresh = self.thresh)
            # print self.feature_type
            features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type)
            all_aus_used = np.array([au.split('_')[0] for au in all_aus])
            idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, idx_test, labels)
        elif self.select_mode=='kunz':
            
            features, labels, all_aus, pain =self.get_feats_by_type(['frequency'], 30,30)

            idx_test, classes_keep_all = Kunz_Selector(type_dataset = self.type_dataset).create_splits(features, labels, all_aus, pain, self.thresh )
            
            features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type)
            all_aus_used = np.array([au.split('_')[0] for au in all_aus])
            
            idx_test_all, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, idx_test, labels)
        else:
            raise ValueError('select_mode '+str(self.select_mode)+' is not valid')
        
        # print all_aus
        # print features.shape
        # print pain

        class_pain = np.in1d(labels, pain).astype(int)
        return features, labels, all_aus, class_pain, idx_test_all, bin_keep_aus

    def select_exp_clinical(self):
        if self.select_mode is None:
            features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type) 
            idx_test, classes_keep_all = self.create_splits(features, labels, all_aus, no_test = True, pain = pain)
            features_clinical, labels_clinical, all_aus_clinical, _ = self.get_feats_by_type(self.feature_type, clinical = True)

            # print all_aus
            # print features.shape
            # print pain
            # print all_aus_clinical

            classes_keep_all = [np.intersect1d(classes_keep_all[0], all_aus_clinical)]
            _, bin_keep_aus = self.convert_vals_to_bins( all_aus, classes_keep_all, [], [])
            _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_clinical, classes_keep_all, [], [])

        elif self.select_mode=='cooc':
            features, labels, all_aus, pain = self.get_feats_by_type(['frequency'])

            idx_test, classes_keep_all = Cooc_Selector(type_dataset = self.type_dataset).create_splits(features, labels, all_aus, pain = pain, thresh = self.thresh, no_test = True)

            # print classes_keep_all, features.shape
            
            features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type)
            features_clinical, labels_clinical, all_aus_clinical,_ = self.get_feats_by_type(self.feature_type, clinical = True)
            # print features.shape, features_clinical.shape
            # raw_input()

            all_aus_used = np.array([au.split('_')[0] for au in all_aus])
            all_aus_used_clinical = np.array([au.split('_')[0] for au in all_aus_clinical])
            classes_keep_all = [np.intersect1d(classes_keep_all[0], all_aus_used_clinical)]            
            
            _, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])
            _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_used_clinical, classes_keep_all, [], [])

        elif self.select_mode=='kunz':

            features, labels, all_aus, pain=self.get_feats_by_type(['frequency'], 30,30)
            idx_test, classes_keep_all = Kunz_Selector(type_dataset = self.type_dataset).create_splits(features, labels, all_aus,pain = pain, thresh = self.thresh, no_test = True)
            # print classes_keep_all, features.shape
            features, labels, all_aus, pain = self.get_feats_by_type(self.feature_type)
            features_clinical, labels_clinical, all_aus_clinical, _ = self.get_feats_by_type(self.feature_type, clinical = True)
            # print features.shape, features_clinical.shape
            # raw_input()

            all_aus_used = np.array([au.split('_')[0] for au in all_aus])
            # print all_aus_used
            # print classes_keep_all
            _, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])
            all_aus_used = np.array([au.split('_')[0] for au in all_aus_clinical])
            _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])


            # all_aus_used = np.array([au.split('_')[0] for au in all_aus])
            # all_aus_used_clinical = np.array([au.split('_')[0] for au in all_aus_clinical])
            # classes_keep_all = [np.intersect1d(classes_keep_all[0], all_aus_used_clinical)]            
            
            # _, bin_keep_aus = self.convert_vals_to_bins( all_aus_used, classes_keep_all, [], [])
            # _, bin_keep_aus_clinical = self.convert_vals_to_bins( all_aus_used_clinical, classes_keep_all, [], [])        
        else:
            raise ValueError('select_mode '+str(self.select_mode)+' is not valid')
        
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
        return train_package, test_package
        