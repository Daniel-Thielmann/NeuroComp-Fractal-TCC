'''
kfold.py

Description:
This module contains the class kfold, which is used to perform a stratified k-fold cross-validation. 
The class is designed to work with eegdata.

Dependencies:
eegdata on modules/core
sklearn
numpy
pandas

'''

from modules.eegdata import eegdata

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import inspect


class kfold:
    ''' kfold object

    Description
    -----------
    This class is used to perform a stratified k-fold cross-validation. 
    The class is designed to work with eegdata.

    Attributes
    ----------
    n_splits : int
        The number of folds to be used in the cross-validation. 
    shuffle : bool
        Whether to shuffle the data before splitting into folds.
    pre_folding : dict
        A dictionary containing the preprocessing functions to be applied to the data before the cross-validation.
        The keys are the names of the preprocessing functions, and the values are tuples containing the function and its parameters.
    pos_folding : dict
        A dictionary containing the postprocessing functions to be applied to the data before the cross-validation.
        The keys are the names of the postprocessing functions, and the values are the functions.
    window_size : float
        The size of the window to be used in the crop method of eegdata.

    Methods
    -------
    __init__(n_splits=5, shuffle=False, pre_folding=None, pos_folding=None, window_size=2.0)
        The constructor of the class.
    exec(target, train_window, test_window)
        Executes the cross-validation.

    '''

    def __init__(self, n_splits=5, shuffle=False, pre_folding=None, pos_folding=None, window_size=2.0):
        ''' Initialize the kfold object.

        Description
        -----------
        The constructor of the class.

        Parameters
        ----------
        n_splits : int
            The number of folds to be used in the cross-validation.
        shuffle : bool
            Whether to shuffle the data before splitting into folds.
        pre_folding : dict
            A dictionary containing the preprocessing functions to be applied to the data before the cross-validation.
            The keys are the names of the preprocessing functions, and the values are tuples containing the function and its parameters.
        pos_folding : dict
            A dictionary containing the postprocessing functions to be applied to the data before the cross-validation.
            The keys are the names of the postprocessing functions, and the values are the functions.
        window_size : float 
            The size of the window to be used in the crop method of eegdata.
        '''
        if type(n_splits) != int:
            raise ValueError("Has to be a int type value")
        else:
            self.n_splits = n_splits
        if type(shuffle) != bool:
            raise ValueError("Has to be a boolean type value")
        else:
            self.shuffle = shuffle
        # check if the pre_folding is a dictionary
        if type(pre_folding) != dict:
            raise ValueError('The pre_folding must be a dictionary.')
        else:
            self.pre_folding = pre_folding
        # check if the pos_folding is a dictionary
        if type(pos_folding) != dict and len(self.pos_folding) > 0:
            raise ValueError('The pos_folding must be a dictionary.')
        else:
            self.pos_folding = pos_folding
        # check if the window_size is a float
        if type(window_size) != float:
            raise ValueError('The window_size must be a float.')
        else:
            self.window_size = window_size

    def exec(self, target, train_window, test_window):
        ''' Execute the cross-validation.

        Description
        -----------
        This method performs the stratified k-fold cross-validation.

        Warning
        -------
        the pos_folding must contain a classifier as the last element of the dictionary.
        Moreover, the function fit_transform must be implemented for pos_folding elements, but the last one.
        the fit_transform method are applied to the training set, and the transform method are applied to the test set.
        Then, if you want to use a method such as Data Augmentation, you must implement the fit_transform method in the class but not the in transform method.

        Parameters
        ----------
        target : eegdata
            The eegdata object containing the data to be used in the cross-validation.
        train_window : list
            A list containing the time windows to be used in the training set. 
        test_window : list
            A list containing the time windows to be used in the test set.

        Returns
        -------
        results : pandas.DataFrame
            A pandas dataframe containing the results of the cross-validation. 
            The columns are 'fold', 'tmin', 'true_label', and the labels of the events in the target object.

        '''

        # check if the target is an eegdata object
        if type(target) != eegdata:
            raise ValueError('The target must be an eegdata object.')

        # check if the train_window and test_window are lists
        if type(train_window) != list or type(test_window) != list:
            raise ValueError('The train_window and test_window must be lists.')
        # check if the train_window and test_window are not empty and all values in train_window is in test_window
        if len(train_window) == 0 or len(test_window) == 0 or not all([t in test_window for t in train_window]):
            raise ValueError(
                'The train_window and test_window must be lists containing at least one time window, and all values in train_window must be in test_window.')

        # check if the pre_folding is a dictionary
        if type(self.pre_folding) != dict:
            raise ValueError('The pre_folding must be a dictionary.')
        # check if the pos_folding is a dictionary
        if type(self.pos_folding) != dict and len(self.pos_folding) > 0:
            raise ValueError('The pos_folding must be a dictionary.')

        # check if the window_size is a float
        if type(self.window_size) != float:
            raise ValueError('The window_size must be a float.')

        # apply the pre_folding to the target object
        target_dict = {}
        for tmin_ in test_window:
            target_dict[tmin_] = target.crop(
                tmin=tmin_, window_size=self.window_size, inplace=False)
            for name, pre_func in self.pre_folding.items():
                if inspect.isfunction(pre_func[0]):
                    target_dict[tmin_] = pre_func[0](
                        target_dict[tmin_], **pre_func[1])
                else:
                    target_dict[tmin_] = pre_func[0].transform(
                        target_dict[tmin_], **pre_func[1])

        y = target.labels
        events_dict = target.labels_dict
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_id = 0
        results = []
        for train_index, test_index in skf.split(y, y):
            fold_id += 1

            if type(target_dict[train_window[0]]) == eegdata:

                target_train = []
                for tmin_ in train_window:
                    target_train.append(
                        target_dict[tmin_].get_data(train_index))

                target_train = eegdata.concatenate(target_train)

                target_test = {}
                for tmin_ in test_window:
                    target_test[tmin_] = target_dict[tmin_].get_data(
                        test_index)

            elif type(target_dict[train_window[0]]) == dict:

                target_train = []
                target_train_labels = []
                for tmin_ in train_window:
                    target_train.append(target_dict[tmin_]['X'][train_index])
                    target_train_labels.append(
                        target_dict[tmin_]['y'][train_index])
                target_train = {'X': np.concatenate(
                    target_train, axis=0), 'y': np.concatenate(target_train_labels, axis=0)}

                target_test = {}
                for tmin_ in test_window:
                    target_test[tmin_] = {
                        'X': target_dict[tmin_]['X'][test_index], 'y': target_dict[tmin_]['y'][test_index]}

            for name, pos_func in self.pos_folding.items():
                if name != 'clf':
                    if inspect.isfunction(pos_func[0]):
                        target_train = pos_func[0](target_train, **pos_func[1])
                    else:
                        target_train = pos_func[0].fit_transform(
                            target_train, **pos_func[1])
                        for tmin_ in test_window:
                            target_test[tmin_] = pos_func[0].transform(
                                target_test[tmin_])
                else:
                    if type(target_train) == eegdata:
                        target_train = {'X': target_train.data,
                                        'y': target_train.labels}

                    clf = pos_func
                    clf[0].fit(target_train['X'], target_train['y'], **clf[1])

            for tmin_ in test_window:
                try:
                    y_pred = clf[0].predict_proba(target_test[tmin_]['X'])
                except:
                    y_pred = np.zeros(
                        (len(target_test[tmin_]['y']), len(events_dict)))
                y_pred = np.round(y_pred, 4)
                for trial_ in range(len(y_pred)):
                    results.append([fold_id, tmin_, find_key_with_value(
                        events_dict, target_test[tmin_]['y'][trial_]), *y_pred[trial_]])

        results = np.array(results)
        results = pd.DataFrame(
            results, columns=['fold', 'tmin', 'true_label', *events_dict.keys()])

        return results


def find_key_with_value(dictionary, i):
    '''find_key_with_value

    Description:
    This function returns the key of a dictionary given a value.

    Parameters:
    dictionary : dict
        The dictionary to be searched.
    i : any
        The value to be searched for.

    Returns:
    key : any
        The key of the dictionary that contains the value i. If the value is not found, returns None.

    '''
    for key, value in dictionary.items():
        if value == i:
            return key
    return None
