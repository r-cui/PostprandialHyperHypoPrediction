import os
from preprocess_utils_ohiot1dm import load_data
from pathlib import Path
import numpy as np
np.random.seed(7)

import warnings
warnings.simplefilter('ignore', Warning)


def extract_data(out_path, mode='train', files='', version='new', patient_index=None):
    sample_rate = 5
    for patient, patient_id in zip(files, patient_index):
        print('\nData exporting for subject {}'.format(patient_id))
        print('Dataset version: ' + version + ', Mode: ' + mode)
        print('Data folder: ' + patient)
        df = load_data(patient)
        df = df.drop("meal_type", axis=1)

        # Data pre-processing will be carried out.
        df['glucose'] = df['glucose'].fillna(-1)
        df['basal'] = df['basal'].fillna(method='ffill')  # fill with previous value
        df["fingerstick"] = df["fingerstick"].fillna(-1)

        df['bolus'] = df['bolus'].fillna(-1)
        df['temp_basal'] = df['temp_basal'].fillna(-1)
        df['basal_end'] = df['basal_end'].fillna(-1)
        df['carbs'] = df['carbs'].fillna(-1)

        df = df.dropna()

        df['index'] = df.index
        df['index_copy'] = df.index

        # remove the duplicate rows by replacing max
        df = df.groupby(df['index_copy']).max()

        # Overwrite basal with temp basal values.
        for i in range(0, len(df)):
            # overwrite basal with temp basal
            if df['temp_basal'][i] != -1:
                basal_timeflag = df['basal_end'][i]
                basal_loop = True
                j = i
                while basal_loop:
                    if df['index'][j] == basal_timeflag:
                        df['basal'][j] = df['temp_basal'][i]
                        basal_loop = False
                    elif df['index'][j] < basal_timeflag:
                        df['basal'][j] = df['temp_basal'][i]
                        j = j + 1
                        if j > len(df) - 1:
                            basal_loop = False
                    elif df['index'][j] > basal_timeflag:
                        basal_loop = False

        # check if duplicates and also missing timestep rows.
        df['gap'] = np.nan
        for i in range(1, len(df)):
            gap = (df['index'][i] - df['index'][i - 1]).total_seconds() / 60.0
            if gap != sample_rate:
                df['gap'][i] = gap
        df['gap'] = df['gap'].fillna(0)

        df.to_csv(os.path.join(out_path, str(patient_id) + '_' + mode + '.csv'), index=False, columns=[
            "index", "glucose", "fingerstick", "basal", "bolus", "carbs", "gap"
        ])
        print('Data export completed for subject {}'.format(patient_id))


def main(ohiot1dm_path, out_path):
    modes_arr = ['train', 'test']
    versions_arr = ['new', 'old']
    for m in modes_arr:
        for v in versions_arr:
            if v == 'old':
                patient_index = [559, 563, 570, 575, 588, 591]
                train_files = ['OhioT1DM-training/559-ws-training.xml', 'OhioT1DM-training/563-ws-training.xml',
                               'OhioT1DM-training/570-ws-training.xml', 'OhioT1DM-training/575-ws-training.xml',
                               'OhioT1DM-training/588-ws-training.xml', 'OhioT1DM-training/591-ws-training.xml']
                file_train = [os.path.join(ohiot1dm_path, s) for s in train_files]
                test_files = ['OhioT1DM-testing/559-ws-testing.xml', 'OhioT1DM-testing/563-ws-testing.xml',
                              'OhioT1DM-testing/570-ws-testing.xml', 'OhioT1DM-testing/575-ws-testing.xml',
                              'OhioT1DM-testing/588-ws-testing.xml', 'OhioT1DM-testing/591-ws-testing.xml']
                file_test = [os.path.join(ohiot1dm_path, s) for s in test_files]
            else:
                patient_index = [540, 544, 552, 567, 584, 596]
                train_files = ['OhioT1DM-2-training/540-ws-training.xml', 'OhioT1DM-2-training/544-ws-training.xml',
                               'OhioT1DM-2-training/552-ws-training.xml', 'OhioT1DM-2-training/567-ws-training.xml',
                               'OhioT1DM-2-training/584-ws-training.xml', 'OhioT1DM-2-training/596-ws-training.xml']
                file_train = [os.path.join(ohiot1dm_path, s) for s in train_files]
                test_files = ['OhioT1DM-2-testing/540-ws-testing.xml', 'OhioT1DM-2-testing/544-ws-testing.xml',
                              'OhioT1DM-2-testing/552-ws-testing.xml', 'OhioT1DM-2-testing/567-ws-testing.xml',
                              'OhioT1DM-2-testing/584-ws-testing.xml', 'OhioT1DM-2-testing/596-ws-testing.xml']
                file_test = [os.path.join(ohiot1dm_path, s) for s in test_files]

            if m == 'train':
                files = file_train
            else:
                files = file_test

            extract_data(out_path, mode=m, files=files, version=v, patient_index=patient_index)


if __name__ == '__main__':
    ohiot1dm_path = "./data/ohiot1dm"
    out_path = os.path.join(ohiot1dm_path, "preprocessed")
    Path(out_path).mkdir(exist_ok=True)
    main(ohiot1dm_path, out_path)
