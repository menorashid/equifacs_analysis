from helpers import util, visualize
import numpy as np
import os
import re

def printable_au(au):
    idx_num = re.search(r'\d', au)
    idx_num = idx_num.start()
    str_au = au[:idx_num].upper()+' '+au[idx_num:]
    return str_au


def cooc_table(bin_keep_aus_all, all_aus_model, coefs = None): 
    bin_keep_aus_all = np.array(bin_keep_aus_all)
    # print bin_keep_aus_all.shape
    # print all_aus_model
    bin_keep_aus_all = bin_keep_aus_all.T
    if coefs is not None:
        coefs = coefs.T
    bin_curr = np.sum(bin_keep_aus_all, axis = 1)>0
    aus_kept = list(all_aus_model[bin_curr])
    aus_kept.sort()
    
    rows = []
    # row = []
    row_title = [' ']+[str(val) for val in [2,5,10,15,20,30]]
    rows.append(row_title)
    for au in aus_kept:
        row = []
        row.append(au)
            # printable_au(au))
        idx_au = np.where(all_aus_model==au)[0][0]
        
        for idx_val,val in enumerate(bin_keep_aus_all[idx_au,:]):
            # print val
            if val:
                if coefs is not None:
                    coef_curr = coefs[idx_au,idx_val]
                # else:
                #     print 'esle'
                    if coef_curr>0:
                        row.append('\checkmark')
                    else:
                        row.append('\mr{\checkmark}') 
                else:
                    row.append('\checkmark')
            else:
                row.append(' ')
        # print ' & '.join(row)
        rows.append(row)

    for row in rows:
        # row = ['{'+val+'}' for val in row]
        print ' & '.join(row)+' \\\\'






def main():
    print 'hello'
    # import re
# >>> s1 = "thishasadigit4here"
# >>> m = re.search(r"\d", s1)
# >>> if m is not None:
# ...     print("Digit found at position", m.start())


if __name__=='__main__':
    main()