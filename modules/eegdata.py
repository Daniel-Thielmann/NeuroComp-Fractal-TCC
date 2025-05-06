'''
eegdata.py

Description
-----------
This module contains the eegdata class which is used to store EEG data and its attributes.
Moreover, it contains the global logger and the set_global_verbose_eegdata function to set the global verbose level for the eegdata module.

Dependencies
------------
numpy
typing
logging
coloredlogs

'''

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np
from typing import Union, List, Optional
import logging
import coloredlogs

# Set the global logger
verbose_levels = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}
global_verbose_eegdata = 'INFO'

def set_global_verbose_eegdata(verbose: str) -> None:
    ''' Set the global verbose level for the eegdata module

    Description
    -----------
    Set the global verbose level for the eegdata module

    Parameters
    ----------
    verbose : str
        Verbose level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    Returns
    -------
    None

    '''

    logging.basicConfig(level=verbose_levels[verbose])
    global_logger = logging.getLogger('global_eegdata')
    global_logger.setLevel(logging.getLevelName(verbose))
    coloredlogs.install(level=verbose_levels[verbose], 
                        logger=global_logger,
                        fmt='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        level_styles={'debug': {'color': 'green'},
                                        'info': {'color': 'green'},
                                        'warning': {'color': 'yellow'},
                                        'error': {'color': 'red'},
                                        'critical': {'color': 'red', 'bold': True}},
                        field_styles={'asctime': {'color': 2},
                                        'levelname': {'color': 'magenta', 'bold': True}})

global_logger = logging.getLogger('global_eegdata')
set_global_verbose_eegdata(global_verbose_eegdata)

class eegdata:
    ''' EEG data object
    
    Description
    -----------
    This class is used to store EEG data and its attributes

    Attributes
    ----------
    id : int
        ID of the object
    data : np.ndarray
        EEG data
    n_trials : int
        Number of trials
    n_bands : int
        Number of frequency bands
    n_channels : int
        Number of channels
    n_times : int
        Number of time points
    sfreq : int
        Sampling frequency
    tmin : float
        Initial time
    timestamps : np.ndarray
        Array of timestamps
    events : dict
        Dictionary of events
    labels : np.ndarray
        Array of labels
    labels_dict : dict
        Dictionary of labels
    ch_names : np.ndarray
        Array of channel names
    local_logger : logging.Logger
        Logger of the object
    verbose : str
        Verbose level

    Methods
    -------
    __init__(self, data, sfreq, labels, labels_dict, events, ch_names, tmin, verbose)
        Initialize the object
    __delete__(self)
        Delete the object
    __str__(self)
        Print the object
    get_data(self, idx)
        Get a new eegdata object with the data indexed by idx
    concatenate(set_of_eeg_data)
        Concatenate a set of eegdata objects
    copy()
        Copy the object
    append(eeg_data_)
        Append an eegdata object to the object
    shuffle()
        Shuffle the data
    crop(tmin, window_size)
        Crop the data
    apply_to_trial(func, used, modified, *args, **kwargs)
        Apply a function to each trial of the data
    apply_to_data(func, used, modified, *args, **kwargs)
        Apply a function to the data

    '''

    def __init__(self, 
                data: np.array, 
                sfreq: int, 
                labels: Optional[Union[None, np.ndarray]] = None,
                labels_dict: Optional[Union[None, dict]] = None,
                events: Optional[dict] = None,
                ch_names: Optional[Union[None, np.ndarray]] = None,
                tmin: Optional[Union[None, float]] = None,
                timestamp: Optional[Union[None, np.ndarray]] = None,
                verbose: Optional[Union[None, str]] = None
                ) -> None:
        ''' Initialize the object
        
        Description
        -----------
        Initialize the object
        
        Parameters
        ----------
        data : np.ndarray
            EEG data
        sfreq : int
            Sampling frequency
        labels : np.ndarray, optional
            Array of labels
        labels_dict : dict, optional
            Dictionary of labels
        events : dict, optional
            Dictionary of events
        ch_names : np.ndarray, optional
            Array of channel names
        tmin : float, optional
            Initial time
        timestamp : np.ndarray, optional
            Array of timestamps
        verbose : str, optional
            Verbose level
        
        Returns
        -------
        None
        
        '''

        # Check and set verbose if it is valid
        if verbose is not None:
            if type(verbose) != str:
                raise ValueError("verbose must be of type str")
            if verbose not in verbose_levels:
                raise ValueError("verbose must be one of {}".format(verbose_levels))
            self._verbose = verbose
        else:
            self._verbose = global_verbose_eegdata
        
        # Set the local logger
        self._local_logger = logging.getLogger('local_eegdata')
        self._local_logger.setLevel(logging.getLevelName(self._verbose))

        # Check that the data is valid
        if type(data) != np.ndarray:
            raise ValueError("data must be an array-like object")        
        elif len(data.shape) > 4:
            raise ValueError("data must be a 4D array-like object or less")
        elif data.dtype != np.float64:
            raise ValueError("data must be of type float64")
        self._id = np.random.randint(0, 1e9)
        if data.ndim == 1:
            self._data = data.reshape(1, 1, 1, data.shape[0])
            self._local_logger.warning(f":\n"
                                 f"\tData with id {self._id} has only one dimension\n"
                                 f"\tAssuming that the data has only one trial, one band and one channel\n"
                                 f"\tplease use a 4D array-like object to avoid this warning\n")
        elif data.ndim == 2:
            self._data = data.reshape(1, 1, data.shape[0], data.shape[1])
            self._local_logger.warning(f":\n"
                                 f"\tData with id {self._id} has only two dimensions\n"
                                 f"\tAssuming that the data has only one trial and one band\n"
                                 f"\tplease use a 4D array-like object to avoid this warning\n")            
        elif data.ndim == 3:
            self._data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
            self._local_logger.warning(f":\n"
                                 f"\tData with id {self._id} has only three dimensions\n"
                                 f"\tAssuming that the data has only one trial\n"
                                 f"\tplease use a 4D array-like object to avoid this warning\n")
        else:
            self._data = data.copy()

        # Check if sfreq is valid
        if type(sfreq) == float and (sfreq - int(sfreq)) == 0:
            sfreq = int(sfreq)
        if type(sfreq) != int or sfreq <= 0:
            raise ValueError("sfreq must be a positive integer")

        # Check that the labels are valid
        if labels is not None:
            if type(labels) == list:
                labels = np.array(labels)
            if type(labels) != np.ndarray:
                raise ValueError("labels must be an array-like object")
            if len(labels.shape) != 1:
                raise ValueError("labels must be a 1D array-like object")           
            if labels.shape[0] != self._data.shape[0]:
                raise ValueError("labels must have the same length as the first dimension of data")
        

        # Check that the labels_dict is valid
            if labels_dict is not None:
                if type(labels_dict) != dict:
                    raise ValueError("labels_dict must be a dict")
                if len(labels_dict) != np.unique(labels).shape[0]:
                    raise ValueError("labels_dict must have the same length as the unique labels")
        else:
            labels = np.zeros(self._data.shape[0])
            labels_dict = {}

        # Check that the events are valid
        if events is not None:
            if type(events) != dict:
                raise ValueError("events must be a dict")
        else:
            events = {}

        # Check that the ch_names are valid
        if ch_names is not None:
            if type(ch_names) == list:
                ch_names = np.array(ch_names)
            if np.ndim(ch_names) != 1:
                raise ValueError("ch_names must be a 1D array-like object")
            if len(ch_names) != self._data.shape[2]:
                raise ValueError("ch_names must have the same length as the third dimension of data")
        else:
            ch_names = np.array([f"ch_{i}" for i in range(self._data.shape[2])])

        if timestamp is not None and tmin is not None:
            raise ValueError("timestamp and tmin cannot be set at the same time")
        elif timestamp is not None:
            if type(timestamp) == list:
                timestamp = np.array(timestamp)
            if type(timestamp) != np.ndarray:
                raise ValueError("timestamp must be an array-like object")
            if len(timestamp.shape) != 1:
                raise ValueError("timestamp must be a 1D array-like object")
            if timestamp.shape[0] != self._data.shape[3]:
                raise ValueError("timestamp must have the same length as the fourth dimension of data")
            tmin = timestamp[0]
        elif tmin is not None:
            if type(tmin) == int:
                tmin = float(tmin)
            if type(tmin) != float:
                raise ValueError("tmin must be of type float")
            timestamp = np.arange(self._data.shape[3]) / sfreq + tmin
        else:
            timestamp = np.arange(self._data.shape[3]) / sfreq
            tmin = timestamp[0]
        # Set the attributes
        
        

        self._n_trials, self._n_bands, self._n_channels, self._n_times = self._data.shape
        self._sfreq = sfreq
        self._tmin = tmin
        self._timestamps = np.arange(self._data.shape[3]) / self._sfreq + tmin
        self._labels = labels
        self._labels_dict = labels_dict
        self._events = events
        self._ch_names = ch_names

        # Log the creation of the object
        self._local_logger.info(f":\n"
                          f"\tCreated eegdata object with id {self._id}\n")
        self._local_logger.debug(f":\n"
                           f"\tid: {self._id}\n"
                           f"\tdata shape: {self._data.shape}\n"
                           f"\ttmin: {tmin}\n"
                           f"\ttimestamps shape: {self._timestamps.shape}\n"
                           f"\tsfreq: {sfreq}\n"
                           f"\tlabels shape: {labels.shape}\n"
                           f"\tlabels_dict: {labels_dict}\n"
                           f"\tevents keys: {events.keys()}\n"
                           f"\tch_names: {ch_names}\n"
                           f"\ttmin: {tmin}\n"
                           f"\tverbose: {verbose}\n")

    def __delete__(self) -> None:
        """ Delete the object
        
        Description
        -----------
        Delete the object and its attributes
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        
        del self._data
        del self._n_trials
        del self._n_bands
        del self._n_channels
        del self._n_times
        del self._sfreq
        del self._tmin
        del self._timestamps
        del self._events
        del self._labels
        del self._labels_dict
        del self._ch_names
        del self._verbose

        self._local_logger.info(f":\n"
                               f"\tDeleted eegdata object with id {self._id}\n") 
        del self._id
        del self._local_logger
        del self

    def __str__(self) -> str:
        """ Print the object

        Description
        -----------
        Print the ID of the object

        Parameters
        ----------
        None

        Returns
        -------
        str
            String representation of the object

        """

        return f"eegdata<id={self._id}>"
    
    # Getters
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def data(self) -> np.ndarray:
        return self._data.copy()
    
    def get_data(self, idx: Union[None, int, np.ndarray]=None, inplace: bool=False) -> 'eegdata':
        ''' Get a new eegdata object with the data indexed by idx

        Description
        -----------
        Get a new eegdata object with the data indexed by idx   

        Parameters
        ----------
        idx : int, np.ndarray, optional
            Index of the data to be retrieved
        inplace : bool, optional
            If True, the data is retrieved in place

        Returns
        -------
        eegdata or None
            New eegdata object with the data indexed by idx or None if inplace is True
        
        '''

        if inplace:
            temp_data = self
        else:
            temp_data = self.copy()
        
        if idx is None:
            return temp_data

        else:
            if type(idx) == int:
                idx = [idx]
            if type(idx) == list:
                idx = np.array(idx).astype(int)
            elif type(idx) == np.ndarray:
                if idx.dtype != int:
                    raise ValueError("idx must be an array-like object of type int")
            else:
                raise ValueError("idx must be an array-like object")
            
            temp_data._data = temp_data._data[idx]
            if temp_data._labels is not None:
                temp_data._labels = temp_data._labels[idx]

            if inplace:
                return None
            else:
                return temp_data

    @property
    def n_trials(self) -> int:
        return self._n_trials
    
    @property
    def n_bands(self) -> int:
        return self._n_bands
    
    @property
    def n_channels(self) -> int:
        return self._n_channels
    
    @property
    def n_times(self) -> int:
        return self._n_times

    @property
    def sfreq(self) -> int:
        return self._sfreq
    
    @property
    def tmin(self) -> float:
        return self._tmin

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps.copy()
    
    @property
    def events(self) -> dict:
        return self._events.copy()
    
    @property
    def labels(self) -> np.ndarray:
        return self._labels.copy()
    
    @property
    def labels_dict(self) -> dict:
        return self._labels_dict.copy()

    @property
    def ch_names(self) -> np.ndarray:
        return self._ch_names.copy()
    
    @property
    def verbose(self) -> str:
        return self._verbose

    # Setters

    @id.setter
    def id(self, id_: int) -> None:
        ''' Set the id
        
        Description
        -----------
        Set the id

        Critical
        -------
        Setting id is not allowed
        After the object is created

        Parameters
        ----------
        id_ : int
            ID of the object

        Returns
        -------
        None
        
        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting id is not allowed\n"
                                    f"\tAfter the object is created\n")
     
    @data.setter
    def data(self, data: np.ndarray) -> None:
        ''' Set the data
        
        Description
        -----------
        set the data

        Warning
        -------
        It is not recommended to set the data directly
        Try to use apply_to_data instead of setting the data directly

        Parameters
        ----------
        data : np.ndarray
            EEG data
        
        Returns
        -------
        None
        
        '''

        # Check that the data is valid
        if type(data) != np.ndarray:
            raise ValueError("data must be an array-like object")
        if data.dtype != np.float64:    
            raise ValueError("data must be of type float64")
        if len(data.shape) > 4:
            raise ValueError("data must be a 4D array-like object or less")
        
        # Set the data
        if data.ndim == 1:
            self._data = data.reshape(1, 1, 1, data.shape[0])
            self._local_logger.warning(f":\n"
                                 f"\tData with id {self._id} has only one dimension\n"
                                 f"\tAssuming that the data has only one trial, one band and one channel\n"
                                 f"\tplease use a 4D array-like object to avoid this warning\n")
        elif data.ndim == 2:
            self._data = data.reshape(1, 1, data.shape[0], data.shape[1])
            self._local_logger.warning(f":\n"
                                 f"\tData with id {self._id} has only two dimensions\n"
                                 f"\tAssuming that the data has only one trial and one band\n"
                                 f"\tplease use a 4D array-like object to avoid this warning\n")            
        elif data.ndim == 3:
            self._data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
            self._local_logger.warning(f":\n"
                                 f"\tData with id {self._id} has only three dimensions\n"
                                 f"\tAssuming that the data has only one trial\n"
                                 f"\tplease use a 4D array-like object to avoid this warning\n")
        else:
            self._data = data.copy()

        self._n_trials, self._n_bands, self._n_channels, self._n_times = self._data.shape
        if self._labels.shape[0] != self._n_trials:
            self._labels = None
            self._local_logger.warning(f":\n"
                                       f"\tData with id {self._id} has a different number of trials than the labels\n"
                                       f"\tSetting labels to None\n")
        if len(self._timestamps) != self._n_times:
            self._timestamps = np.arange(self._data.shape[3]) / self._sfreq
            self._tmin = 0.0
            self._local_logger.warning(f":\n"
                                       f"\tData with id {self._id} has a different number of time points than the timestamps\n"
                                       f"\tSetting timestamps to np.arange(data.shape[3]) / sfreq\n")

        # Log the setting of the data
        self._local_logger.info(f":\n"
                                f"\tSet data of eegdata object with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                 f"\tid: {self._id}\n"
                                 f"\tnew data shape: {data.shape}\n"
                                 f"\tnew timestamps shape: {self._timestamps.shape}\n")
        self._local_logger.warning(f":\n"
                                   f"\tIt is not recommended to set the data directly\n"
                                   f"\tTry to use apply_to_data instead of setting the data directly\n")

    @n_trials.setter
    def n_trials(self, n_trials: int) -> None:
        ''' Set the number of trials

        Description
        -----------
        Set the number of trials

        Critical
        -------
        Setting n_trials is not allowed
        The new data must be set instead

        Parameters
        ----------
        n_trials : int
            Number of trials
        
        Returns
        -------
        None

        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting n_trials is not allowed\n"
                                    f"\tThe new data must be set instead\n")
        
    @n_bands.setter
    def n_bands(self, n_bands: int) -> None:
        ''' Set the number of bands
        
        Description
        -----------
        Set the number of bands
        
        Critical
        -------
        Setting n_bands is not allowed
        The new data must be set instead
        
        Parameters
        ----------
        n_bands : int
        Number of bands
        
        Returns
        -------
        None
        
        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting n_bands is not allowed\n"
                                    f"\tThe new data must be set instead\n")
    
    @n_channels.setter
    def n_channels(self, n_channels: int) -> None:
        ''' Set the number of channels

        Description
        -----------
        Set the number of channels

        Critical
        -------
        Setting n_channels is not allowed
        The new data must be set instead

        Parameters
        ----------
        n_channels : int
            Number of channels

        Returns
        -------
        None
        
        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting n_channels is not allowed\n"
                                    f"\tThe new data must be set instead\n")
        
    @n_times.setter
    def n_times(self, n_times: int) -> None:
        ''' Set the number of time points

        Description
        -----------
        Set the number of time points

        Critical
        -------
        Setting n_times is not allowed
        The new data must be set instead

        Parameters
        ----------
        n_times : int
            Number of time points

        Returns
        -------
        None

        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting n_times is not allowed\n"
                                    f"\tThe new data must be set instead\n")

    @sfreq.setter
    def sfreq(self, sfreq: int) -> None:
        ''' Set the sampling frequency

        Description
        -----------
        Set the sampling frequency

        Critical
        -------
        Setting sampling frequency is not allowed
        If you want to change the sampling frequency
        Please use resample or apply_to_data instead

        Parameters
        ----------
        sfreq : int
            Sampling frequency

        Returns
        -------
        None

        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting sample frequency is not allowed\n"
                                    f"\tIf you want to change the sampling frequency\n"
                                    f"\tPlease use resample or apply_to_data instead\n")

    @tmin.setter
    def tmin(self, tmin: float) -> None:
        ''' Set the initial time

        Description
        -----------
        Set the initial time

        Critical
        -------
        Setting tmin is not allowed
        If you want to change the tmin
        Please use shift_timestamps instead

        Parameters
        ----------
        tmin : float
            Initial time

        Returns
        -------
        None

        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting tmin is not allowed\n"
                                    f"\tIf you want to change the tmin\n"
                                    f"\tPlease use shift_timestamps instead\n")

    @timestamps.setter
    def timestamps(self, timestamps: np.ndarray) -> None:
        ''' Set the timestamps

        Description
        -----------
        Set the timestamps

        Critical
        -------
        Setting timestamps is not allowed
        Please use shift_timestamps instead
        If you want to change the sampling frequency too
        please use apply_to_data instead

        Parameters
        ----------
        timestamps : np.ndarray
            Array of timestamps

        Returns
        -------
        None

        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting timestamps is not allowed\n"
                                    f"\tPlease use shift_timestamps instead\n"
                                    f"\tIf you want to change the sampling frequency too\n"
                                    f"\tplease use apply_to_data instead\n")

    @events.setter
    def events(self, events: dict) -> None:
        ''' Set the events
        
        Description
        -----------
        
        Parameters
        ----------
        events : dict
            Dictionary of events
            
        Returns
        -------
        None
            
        '''
        
        # Check that the events are valid
        if type(events) != dict:
            raise ValueError("events must be a dict")
        
        # Set the events
        self._events = events.copy()
        
        # Log the setting of the events
        self._local_logger.info(f":\n"
                                f"\tSet events of eegdata object with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                 f"\tid: {self._id}\n"
                                 f"\tevents keys: {events.keys()}\n")
        
    @labels.setter
    def labels(self, labels: np.ndarray) -> None:
        ''' Set the labels
        
        Description
        -----------
        
        Parameters
        ----------
        labels : np.ndarray
            Array of labels
            
        Returns
        -------
        None
            
        '''
        
        # Check that the labels are valid
        if type(labels) != np.ndarray:
            raise ValueError("labels must be an array-like object")
        if len(labels.shape) != 1:
            raise ValueError("labels must be a 1D array-like object")
        if labels.shape[0] != self._data.shape[0]:
            raise ValueError("labels must have the same length as the first dimension of data")
        
        # Set the labels
        self._labels = labels.copy()

        # Log the setting of the labels
        self._local_logger.info(f":\n"
                                f"\tSet labels of eegdata object with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                 f"\tid: {self._id}\n"
                                 f"\tlabels shape: {labels.shape}\n")

    @labels_dict.setter
    def labels_dict(self, labels_dict: dict) -> None:
        ''' Set the labels_dict
        
        Description
        -----------
        
        Parameters
        ----------
        labels_dict : dict
            Dictionary of labels
            
        Returns
        -------
        None
            
        '''
        
        # Check that the labels_dict is valid
        if type(labels_dict) != dict:
            raise ValueError("labels_dict must be a dict")
        if len(labels_dict) != np.unique(self._labels).shape[0]:
            raise ValueError("labels_dict must have the same length as the unique labels")
        
        # Set the labels_dict
        self._labels_dict = labels_dict.copy()

        # Log the setting of the labels_dict
        self._local_logger.info(f":\n"
                                f"\tSet labels_dict of eegdata object with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                 f"\tid: {self._id}\n"
                                 f"\tlabels_dict: {labels_dict}\n")

    @ch_names.setter
    def ch_names(self, ch_names: np.ndarray) -> None:
        ''' Set the channel names
        
        Description
        -----------
        
        Parameters
        ----------
        ch_names : np.ndarray
            Array of channel names
            
        Returns
        -------
        None
            
        '''
        
        # Check that the ch_names are valid
        if np.ndim(ch_names) != 1:
            raise ValueError("ch_names must be a 1D array-like object")
        if len(ch_names) != self._data.shape[2]:
            raise ValueError("ch_names must have the same length as the third dimension of data")
        
        # Set the ch_names
        self._ch_names = ch_names.copy()

        # Log the setting of the ch_names
        self._local_logger.info(f":\n"
                                f"\tSet ch_names of eegdata object with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                 f"\tid: {self._id}\n"
                                 f"\tch_names: {ch_names}\n")
    
    @verbose.setter
    def verbose(self, verbose: str) -> None:
        ''' Set the verbose level

        Description
        -----------
        Set the verbose level

        Critical
        -------
        Setting verbose is not allowed
        After the object is created

        Parameters
        ----------
        verbose : str
            Verbose level

        Returns
        -------
        None

        '''
        self._local_logger.critical(f":\n"
                                    f"\tSetting verbose is not allowed\n"
                                    f"\tAfter the object is created\n")

    # Methods

    @staticmethod
    def concatenate(set_of_eeg_data: Union[List['eegdata'], 'eegdata']) -> 'eegdata':
        ''' Concatenate a set of eegdata objects

        Description
        -----------
        Concatenate a set of eegdata objects

        Parameters
        ----------
        set_of_eeg_data : list of eegdata objects
            Set of eegdata objects to be concatenated or a single eegdata object

        Returns
        -------
        eegdata
            Concatenated eegdata object

        '''

        # Check that the set of eegdata objects is valid
        if type(set_of_eeg_data) != list and type(set_of_eeg_data) != eegdata:
            raise ValueError("set_of_eeg_data must be a list of eegdata objects or a single eegdata object")
        if type(set_of_eeg_data) == eegdata:
            set_of_eeg_data = [set_of_eeg_data]
        if len(set_of_eeg_data) == 0:
            raise ValueError("set_of_eeg_data must not be empty")
        for eeg_data_ in set_of_eeg_data:
            if type(eeg_data_) != eegdata:
                raise ValueError("set_of_eeg_data must contain only eegdata objects")
        
        # verify that the eegdata objects have the same sampling frequency, number of channels and number of bands, tmin
        sfreq = set_of_eeg_data[0].sfreq
        n_channels = set_of_eeg_data[0].data.shape[2]
        n_bands = set_of_eeg_data[0].data.shape[1]
        tmin = set_of_eeg_data[0].timestamps[0]
        ch_names = set_of_eeg_data[0].ch_names
        events = set_of_eeg_data[0].events
        for eeg_data_ in set_of_eeg_data[1:]:
            if eeg_data_.sfreq != sfreq:
                raise ValueError("all eegdata objects must have the same sampling frequency")
            if eeg_data_.data.shape[2] != n_channels:
                raise ValueError("all eegdata objects must have the same number of channels")
            if eeg_data_.data.shape[1] != n_bands:
                raise ValueError("all eegdata objects must have the same number of bands")
            if not np.array_equal(eeg_data_.ch_names, ch_names):
                raise ValueError("all eegdata objects must have the same channel names")
            if not eeg_data_.events == events:
                raise ValueError("all eegdata objects must have the same events")
        
        
        # verify tmin is the same for all eegdata objects, if not shift the timestamps and set tmin to zero for all eegdata objects
        all_same_tmin = True
        for eeg_data_ in set_of_eeg_data:
            if eeg_data_.tmin != tmin:
                all_same_tmin = False
                global_logger.warning(f":\n"
                                      f"\tData with id {eeg_data_._id} has a different tmin than the other data\n"
                                      f"\tShifting timestamps and setting tmin to zero\n")
                
                break
        
        if not all_same_tmin:
            tmin = 0.0
        else:
            tmin = set_of_eeg_data[0].tmin

        data = []
        labels = []
        for eeg_data_ in set_of_eeg_data:
            data.append(eeg_data_.data)
            for labels_ in eeg_data_.labels:
                labels.append(find_key_with_value(eeg_data_.labels_dict, labels_))

        data = np.concatenate(data)
        #Problema do relacionamento labels / dict_labels quando n tem, tem mts, tem do mesmo jeito e de maneiras similares.
        unique = np.unique(labels)
        labels_dict = {unique[i]: i+1 for i in range(unique.shape[0])}
        labels = np.array([labels_dict[label] for label in labels])
        
        # create the new eegdata object
        eeg_data_ = eegdata(data=data, 
                            sfreq=sfreq, 
                            labels=labels, 
                            events=events, 
                            ch_names=ch_names,
                            labels_dict = labels_dict,
                            tmin=tmin,
                            verbose=eeg_data_.verbose)

        # Log the concatenation
        global_logger.info(f":\n"
                           f"\tConcatenated data with id {eeg_data_._id}\n")
        global_logger.debug(f":\n"                           
                            f"\tid: {eeg_data_._id}\n"
                            f"\tnew data shape: {data.shape}\n"
                            f"\tnew labels shape: {labels.shape}\n")
        
        return eeg_data_

    def copy(self) -> 'eegdata':
        ''' Copy the object
        
        Description
        -----------
        Copy the object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        eegdata
            Copied eegdata object

        '''

        self._local_logger.info(f":\n"
                               f"\tCopying data with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                f"\tdata shape: {self._data.shape}\n"
                                f"\tsfreq: {self._sfreq}\n"
                                f"\ttmin: {self._tmin}\n"
                                f"\ttimestamps shape: {self._timestamps.shape}\n"
                                f"\tevents keys: {self._events.keys()}\n"
                                f"\tlabels shape: {self.labels.shape}\n"
                                f"\tch_names: {self._ch_names}\n"
                                f"\tverbose: {self._verbose}\n")
        
        return eegdata(data = self.data, 
                       sfreq = self.sfreq, 
                       labels = self.labels,
                       labels_dict = self.labels_dict,
                       events = self.events, 
                       ch_names = self.ch_names, 
                       tmin = self.tmin,
                       verbose = self.verbose)
 
    def append(self, eeg_data_: 'eegdata') -> None:
        ''' Append an eegdata object to the object

        Description
        -----------
        Append an eegdata object to the object

        Parameters
        ----------
        eeg_data_ : eegdata
            eegdata object to be appended

        Returns
        -------
        None

        '''

        # Check that the eegdata object is valid
        if type(eeg_data_) != eegdata:
            raise ValueError("eeg_data_ must be an eegdata object")

        # verify that the eegdata objects have the same sampling frequency, number of channels and number of bands, tmin
        if eeg_data_.sfreq != self.sfreq:
            raise ValueError("eegdata object must has the same sampling frequency")
        if eeg_data_.data.shape[2] != self.data.shape[2]:
            raise ValueError("eegdata object must has the same number of channels")
        if eeg_data_.data.shape[1] != self.data.shape[1]:
            raise ValueError("eegdata object must has the same number of bands")
            
        # verify tmin is the same for all eegdata objects, if not shift the timestamps and set tmin to zero for all eegdata objects
        if eeg_data_.tmin != self.tmin:
            global_logger.warning(f":\n"
                                  f"\tData with id {eeg_data_._id} has a different tmin than the other data\n"
                                    f"\tUsing the tmin of the original data\n")

        # Concatenate the data and labels
        data = np.concatenate((self.data, eeg_data_.data))
        labels = np.concatenate((self.labels, eeg_data_.labels))

        # subscribe to the new data
        self._data = data
        self._labels = labels

        # Log the concatenation
        self._local_logger.info(f":\n"
                               f"\tAppended data with id {eeg_data_._id} to data with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                f"\tid: {self._id}\n"
                                f"\tnew data shape: {data.shape}\n"
                                f"\tnew labels shape: {labels.shape}\n")
   
    def shuffle(self) -> None:
        ''' Shuffle the data

        Description
        -----------
        Shuffle the data
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        
        '''

        # shuffle indices
        indices = np.arange(self._data.shape[0])
        np.random.shuffle(indices)
        
        # shuffle data
        self._data = self._data[indices]

        # shuffle labels
        if self.labels is not None:
            self.labels = self.labels[indices]

        # Log the shuffling
        self._local_logger.info(f":\n"
                               f"\tShuffled data with id {self._id}\n")
     
    def crop(self, tmin: Union[None, float, np.ndarray, list], window_size: float, inplace: bool = True) -> None:
        ''' Crop the data
        
        Description
        -----------
        Crop the data
        
        Parameters
        ----------
        tmin : float or np.ndarray, optional
            Initial time
        window_size : float
            Window size
        
        Returns
        -------
        None
        
        '''

        # Check that the tmin is valid
        if tmin is not None:
            if type(tmin) == int:
                tmin = float(tmin)
            if type(tmin) == float or type(tmin) == np.float64:
                tmin = np.array([tmin])
            if type(tmin) == list:
                tmin = np.array(tmin)
            if type(tmin) != np.ndarray:
                print(type(tmin))
                raise ValueError("tmin must be of type float or np.ndarray")
            if type(tmin) == np.ndarray:
                if tmin.ndim != 1:
                    raise ValueError("tmin must be a 1D array-like object")
        # falta caso para tratar tmin == None.
                
        # Check that the window is valid
        if type(window_size) == int:
            window_size = float(window_size)
        if type(window_size) != float:
            raise ValueError("window_size must be of type float")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self = self if inplace else self.copy()

        # Crop the data
        new_data = []
        new_labels = []
        for tmin_ in tmin:

            # Find the indices of the first sample
            indices = ((tmin_ - self._timestamps[0]) * self._sfreq).astype(int)
            max_indices = indices + int(window_size * self._sfreq)
            if np.any(indices + int(window_size * self._sfreq) > self._data.shape[3]):
                raise ValueError("tmin + window_size must be less than or equal to the tmax of the original data")

            # Crop the data
            new_data.append(self._data[:, :, :, indices:max_indices])
            if self.labels is not None:
                new_labels.append(self.labels)

        # Concatenate the data
        self._data = np.concatenate(new_data)
        if self.labels is not None:
            self.labels = np.concatenate(new_labels)
        self._n_trials, self._n_bands, self._n_channels, self._n_times = self._data.shape

        # update timestamps
        if len(tmin) == 1:
            self._timestamps = np.arange(tmin[0], tmin[0] + window_size, 1 / self._sfreq)
        else:
            self._timestamps = np.arange(0, window_size, 1 / self._sfreq)

        # Log the cropping
        self._local_logger.info(f":\n"
                                f"\tCropped data with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                f"\tnew data shape: {self._data.shape}\n"
                                f"\tnew timestamps shape: {self._timestamps.shape}\n"
                                f"\tnew tmin: {self._timestamps[0]}\n")
        
        if not inplace:
            return self
                                 
    def apply_to_trial(self, func, attributes=[], *args, **kwargs) -> None:
        ''' Apply a function to each trial of the data
        
        Description
        -----------
        Apply a function to each trial of the data
        
        Parameters
        ----------
        func : function
            Function to be applied  
        attributes : list, optional
            List of attributes to be used by the function
        args : tuple, optional
            Positional arguments to be passed to the function
        kwargs : dict, optional
            Keyword arguments to be passed to the function
            
        Returns
        -------
        None
        
        '''

        # verify if object has __self__.__class__
        if hasattr(func, '__self__') and hasattr(func.__self__, '__class__'):        
            func_name = func.__self__.__class__.__name__ + '.' + func.__name__
        else:
            func_name = func.__name__

        #verify if used and modified is in the possible attributes
        possible_attributes = ['sfreq', 'tmin', 'timestamps', 'events', 'ch_names']
        for attribute in attributes:
            if attribute not in possible_attributes:
                raise ValueError(f"{attribute} is not a valid attribute")

        # create a dictionary of the used attributes to use in func
        attributes_ = {}
        for attribute in attributes:
            attributes_[attribute] = getattr(self, attribute)
            if type(attributes_[attribute]) == np.ndarray or type(attributes_[attribute]) == dict:
                attributes_[attribute] = attributes_[attribute].copy()
            else:
                attributes_[attribute] = getattr(self, attribute)

        # Apply the function to each trial of the data
        data_ = []
        for trial_ in range(self._data.shape[0]):
            exit_ = func(self._data[trial_].copy(), *args, **attributes_, **kwargs)
            if type(exit_) != dict:
                data_.append(exit_)
            else:
                data_.append(exit_['data'])

        self._data = np.array(data_)
        self._n_trials, self._n_bands, self._n_channels, self._n_times = self._data.shape

        #if dict has sfreq, tmin, timestamps, events, ch_names, update them
        if type(exit_) == dict:
            if 'sfreq' in exit_.keys():
                self._sfreq = exit_['sfreq']
                self._timestamps = np.arange(self._data.shape[3]) / self._sfreq
            if 'events' in exit_.keys():
                self._events = exit_['events']
            if 'ch_names' in exit_.keys():
                self._ch_names = exit_['ch_names']

            if 'timestamps' and 'tmin' in exit_.keys():
                self._local_logger.warning(f":\n"
                                             f"\tBoth timestamps and tmin were modified\n"
                                             f"\tPlease check if this was intended\n"
                                             f"Only timestamps will be used\n")
            
            if 'timestamps' in exit_.keys():
                self._timestamps = exit_['timestamps']
            elif 'tmin' in exit_.keys():
                self.timestamps = np.arange(self._data.shape[3]) / self._sfreq + exit_['tmin']
        
        # Log the application
        self._local_logger.info(f":\n"
                               f"\tApplied function to each trial of data with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                f"\tfunction: {func_name}\n"
                                f"\targs: {args}\n"
                                f"\tkwargs: {kwargs}\n"
                                f"\tattributes: {attributes}\n"
                                f"\tnew data shape: {self._data.shape}\n")
        
        return self

    def apply_to_data(self, func, attributes=[], *args, **kwargs) -> None:
        ''' Apply a function to the data
        
        Description
        -----------
        Apply a function to the data
        
        Parameters
        ----------
        func : function
            Function to be applied
            
        attributes : list, optional
            List of attributes to be used by the function
        args : tuple, optional
            Positional arguments to be passed to the function
        kwargs : dict, optional
            Keyword arguments to be passed to the function

        Returns
        -------
        None

        '''

        # verify if object has __self__.__class__
        if hasattr(func, '__self__') and hasattr(func.__self__, '__class__'):        
            func_name = func.__self__.__class__.__name__ + '.' + func.__name__
        else:
            func_name = func.__name__

        #verify if used and modified is in the possible attributes
        possible_attributes = ['labels', 'sfreq', 'tmin', 'timestamps', 'events', 'ch_names']
        for attribute in attributes:
            if attribute not in possible_attributes:
                raise ValueError(f"{attribute} is not a valid attribute")

        # create a dictionary of the used attributes to use in func
        attributes_ = {}
        for attribute in attributes:
            attributes_[attribute] = getattr(self, attribute)
            if type(attributes_[attribute]) == np.ndarray or type(attributes_[attribute]) == dict:
                attributes_[attribute] = attributes_[attribute].copy()
            else:
                attributes_[attribute] = getattr(self, attribute)

        # Apply the function to the data
        exit_ = func(self._data.copy(), *args, **attributes_, **kwargs)
        if type(exit_) != dict:
            data_ = exit_
        else:
            data_ = exit_['data']

        self._data = np.array(data_)
        self._n_trials, self._n_bands, self._n_channels, self._n_times = self._data.shape

        #if dict has sfreq, tmin, timestamps, events, ch_names, update them
        if type(exit_) == dict:
            if 'labels' in exit_.keys():
                self._labels = exit_['labels']
            if 'sfreq' in exit_.keys():
                self._sfreq = exit_['sfreq']
            if 'events' in exit_.keys():
                self._events = exit_['events']
            if 'ch_names' in exit_.keys():
                self._ch_names = exit_['ch_names']

            if 'timestamps' and 'tmin' in exit_.keys():
                self._local_logger.warning(f":\n"
                                             f"\tBoth timestamps and tmin were modified\n"
                                             f"\tPlease check if this was intended\n"
                                             f"Only timestamps will be used\n")
            
            if 'timestamps' in exit_.keys():
                self._timestamps = exit_['timestamps']
            elif 'tmin' in exit_.keys():
                self.timestamps = np.arange(self._data.shape[3]) / self._sfreq + exit_['tmin']
        
        # Log the application
        self._local_logger.info(f":\n"
                               f"\tApplied function to each trial of data with id {self._id}\n")
        self._local_logger.debug(f":\n"
                                f"\tfunction: {func_name}\n"
                                f"\targs: {args}\n"
                                f"\tkwargs: {kwargs}\n"
                                f"\tattributes: {attributes}\n"
                                f"\tnew data shape: {self._data.shape}\n")    

        return self

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
