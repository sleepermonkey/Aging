"""
This file generates the sample_submission.csv baseline.
Ensure that the 'public_data' is in the same directory
as this file.
"""

import pandas as pd
import json
import numpy as np
import math
import time
import datetime

import os


def brier_score(target, predicted, class_weights):
    return np.power(target - predicted, 2.0).dot(class_weights).mean()


def load_metadata():
    annotation_names = json.load(open(os.path.join('../public_data', 'annotations.json')))
    location_names = ('bath', 'bed1', 'bed2', 'hall', 'kitchen', 'living', 'stairs', 'study', 'toilet')
    return annotation_names,location_names


def load_accelerations(directory):
    acceleration = pd.read_csv(os.path.join('../public_data', 'train', directory, 'acceleration.csv'))
    acceleration.fillna(0, inplace=True)
    acceleration['a_ascend'] = 0
    acceleration['a_descend'] = 0
    acceleration['a_jump'] = 0
    acceleration['a_loadwalk'] = 0
    acceleration['a_walk'] = 0
    acceleration['p_bent'] = 0
    acceleration['p_kneel'] = 0
    acceleration['p_lie'] = 0
    acceleration['p_sit'] = 0
    acceleration['p_squat'] = 0
    acceleration['p_stand'] = 0
    acceleration['t_bend'] = 0
    acceleration['t_kneel_stand'] = 0
    acceleration['t_lie_sit'] = 0
    acceleration['t_sit_lie'] = 0
    acceleration['t_sit_stand'] = 0
    acceleration['t_stand_kneel'] = 0
    acceleration['t_stand_sit'] = 0
    acceleration['t_straighten'] = 0
    acceleration['t_turn'] = 0

    return acceleration


def load_annotations(directory):
    annotation = pd.DataFrame()
    list_ = []

    for i in range(0,2):
        if os.path.isfile(os.path.join('../public_data', 'train', directory, 'annotations_' + str(i) + '.csv')):
            df = pd.read_csv(os.path.join('../public_data', 'train', directory, 'annotations_' + str(i) + '.csv'))
            list_.append(df)
    annotation = pd.concat(list_)
    annotation = annotation.sort_values(by=['start','end'], ascending=[True,True])
    annotation = annotation.reset_index(drop=True)

    return annotation


def load_locations(directory):
    location = pd.DataFrame()
    list_ = []

    for i in range(0,2):
        if os.path.isfile(os.path.join('../public_data', 'train', directory, 'location_' + str(i) + '.csv')):
            df = pd.read_csv(os.path.join('../public_data', 'train', directory, 'location_' + str(i) + '.csv'))
            list_.append(df)
        location = pd.concat(list_)

    return location


def train_model(annotation_names,pir_locations):
    prior_probs = np.zeros(len(annotation_names))

    for ii in range(1, 2):
        meta = json.load(open(os.path.join('../public_data', 'train', str(ii).zfill(5), 'meta.json')))
        starts = range(math.ceil(meta['end']))
        ends = range(1, math.ceil(meta['end']) + 1)

        pir = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'pir.csv'))
        vid_hallway = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'video_hallway.csv'))
        vid_kitchen = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'video_kitchen.csv'))
        vid_living = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'video_living_room.csv'))

        annotation = load_annotations(str(ii).zfill(5))
        #location = load_locations(str(ii).zfill(5))

        acceleration = load_accelerations(str(ii).zfill(5))

        for start, end in zip(starts, ends):
            df_period = annotation[annotation.start.between(start, end) == True]
            if df_period.start.count() > 0:
                for index, row in df_period.iterrows():
                    acceleration.loc[acceleration.t.between(start, end) == True,annotation_names[row['index']]] += 1

        df = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'targets.csv'))

        non_nans = df[df.isnull().any(axis=1) == False]
        prior_probs += np.asarray(non_nans.mean(axis=0)[annotation_names].tolist())

    prior_probs /= prior_probs.sum()

    weight = json.load(open(os.path.join('../public_data', 'class_weights.json')))
    TotalScore = 0
    for ii in range(1, 2):
        df = pd.read_csv(os.path.join('../public_data', 'train', str(ii).zfill(5), 'targets.csv'))
        df = df.drop(['start', 'end'], axis=1)
        print(str(ii) + ' ' + str(brier_score(df,prior_probs,weight)))
        TotalScore += brier_score(df,prior_probs,weight)

    print('Total Score : ' + str(TotalScore/10.))



start_time = time.time()

annotation_names,location_names = load_metadata()
train_model(annotation_names,location_names)

elapsed = (time.time() - start_time)
print('Task completed in:', datetime.timedelta(seconds=elapsed))
#os.system('say "Finished"')