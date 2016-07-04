import pandas as pd
import json
import numpy as np

import os


def useful_script():
    """
    Load the required metadata
    """
    annotation_names = json.load(open(os.path.join('../public_data', 'annotations.json')))

    """
    Calculate training training class distribution
    """

    prior_probs = np.zeros(len(annotation_names))

    for ii in range(1, 11):
        df = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'targets.csv'))

        non_nans = df[df.isnull().any(axis=1) == False]
        prior_probs += np.asarray(non_nans.mean(axis=0)[annotation_names].tolist())

    prior_probs /= prior_probs.sum()
    prior_probs = prior_probs.tolist()

    """
    Generate submission file
    """

    se_cols = ['start', 'end']

    with open(os.path.join('../public_data', 'sample_submission.csv'), 'w') as fil:
        for te_ind_str in sorted(os.listdir(os.path.join('../public_data', 'test'))):
            if te_ind_str.startswith('.') and os.path.isfile(os.path.join('../public_data', te_ind_str)):
                continue
            te_ind = int(te_ind_str)

            meta = json.load(open(os.path.join('../public_data', 'test', te_ind_str, 'meta.json')))

            starts = range(meta['end'])
            ends = range(1, meta['end'] + 1)

            for start, end in zip(starts, ends):
                row = [te_ind, start, end] + prior_probs

                fil.write(','.join(map(str, row)))
                fil.write('\n')

