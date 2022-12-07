from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import h5py
import torch
import numpy as np
import itertools
from torch.utils.data import Dataset,DataLoader

class H5DatasetSK(Dataset):
    
    def __init__(self, data_dirs, transform=None, flavour=None, limit_num_files=0, start_fraction=0., use_fraction=1.0, read_keys=[]):
        """
        Args: data_dirs ... a list of data directories to find files (up to 10 files read from each dir)
              transform ... a function applied to pre-process data 
              flavour ..... a string that is required to be present in the filename
              limit_num_files ... an integer limiting number of files to be taken per data directory 
              start_fraction .... a floating point fraction (0.0=>1.0) to specify which entry to start reading (per file)
              use_fraction ...... a floating point fraction (0.0=>1.0) to specify how much fraction of a file to be read out (per file)
              read_keys ......... a list of string values = data product keys in h5 file to be read-in (besides 'event_data' and 'labels')
        """
        self._transform = transform
        self._files = []

        # Check input fractions makes sense
        assert start_fraction >= 0. and start_fraction < 1.
        assert use_fraction > 0. and use_fraction <= 1.
        assert (start_fraction + use_fraction) <= 1.
        
        # Load files (up to 10) from each directory in data_dirs list
        for d in data_dirs:
            file_list = [ os.path.join(d,f) for f in os.listdir(d) if flavour is None or flavour in f ]
            if limit_num_files: file_list = file_list[0:limit_num_files]
            self._files += file_list

        print("FILES!")
        print(self._files)
            
        # Create a list of keys.
        f = h5py.File(self._files[0],mode='r')
        assert 'event_data_barrel' in f.keys() and 'nhit_barrel' in f.keys() and 'hit_index_barrel' in f.keys()
        assert 'event_data_top' in f.keys() and 'nhit_top' in f.keys() and 'hit_index_top' in f.keys()
        assert 'event_data_bottom' in f.keys() and 'nhit_bottom' in f.keys() and 'hit_index_bottom' in f.keys()
        self._keys = ['labels', 'nhit_barrel', 'hit_index_barrel', 'event_data_barrel', 
                      'nhit_top', 'hit_index_top', 'event_data_top',
                      'nhit_bottom', 'hit_index_bottom', 'event_data_bottom']
        for key in read_keys:
            if not key in f.keys():
                print('Key',key,'not found in h5 file',self._files[0])
                raise ValueError
            self._keys.append(key)

        # Loop over files and scan events
        self._event_to_file_index  = []
        self._event_to_entry_index = []
        for file_index, file_name in enumerate(self._files):
            f = h5py.File(file_name,mode='r')
            data_size = f['labels'].shape[0]
            start_entry = int(start_fraction * data_size)
            num_entries = int(use_fraction * data_size)
            self._event_to_file_index += [file_index] * num_entries
            self._event_to_entry_index += range(start_entry, start_entry+num_entries)
            f.close()
        
    def __len__(self):
        return len(self._event_to_file_index)
    
    def __getitem__(self,idx):        
        file_index = self._event_to_file_index[idx]
        entry_index = self._event_to_entry_index[idx]
        fh = h5py.File(self._files[file_index], mode='r')
        h5_dict = {}            
        #print(idx, fh['nhit_barrel'][entry_index][0], fh['hit_index_barrel'][entry_index][0])
        for key in self._keys:
            #h5_dict.update({key: []})
            if key == 'event_data_barrel':
                if (fh['nhit_barrel'][entry_index][0]) > 0:
                    h5_dict.update({key: fh[key][fh['hit_index_barrel'][entry_index][0]:fh['hit_index_barrel'][entry_index][0]+fh['nhit_barrel'][entry_index][0]].tolist()})
            elif key == 'event_data_top':
                if (fh['nhit_top'][entry_index]) > 0:                    
                    h5_dict.update({key: fh[key][fh['hit_index_top'][entry_index][0]:fh['hit_index_top'][entry_index][0]+fh['nhit_top'][entry_index][0]][0].tolist()})
            elif key == 'event_data_bottom':
                if (fh['nhit_bottom'][entry_index][0]) > 0:                                
                    h5_dict.update({key: fh[key][fh['hit_index_bottom'][entry_index][0]:fh['hit_index_bottom'][entry_index][0]+fh['nhit_bottom'][entry_index][0]].tolist()})
            else:
                h5_dict.update({key: fh[key][entry_index][0]})
                
        h5_dict.update({'Input_index': idx})
        h5_dict.update({'Entry_index': entry_index})
        #print(h5_dict)
        return h5_dict

def Collate(batch):

    dict_keys = list(batch[0].keys())
    #print(dict_keys)
    #barrel_index = dict_keys.index('event_data_barrel')
    #top_index = dict_keys.index('event_data_top')
    #bottom_index = dict_keys.index('event_data_bottom')

    result = [[] for i in range(len(dict_keys))]    
    for i,data in enumerate(batch):
        for j in range(len(dict_keys)):
            result[j].append(data[dict_keys[j]])
 
    #for i in range(len(result[barrel_index])):
    #    print(result[barrel_index][i].shape, result[top_index][i].shape, result[bottom_index][i].shape)
    
    return(result)
    # [ label, nhit_barrel, hit_index_barrel, event_data_barrel, nhit_top, hit_index_top, event_data_top, nhit_bottom, hit_index_bottom, event_data_bottom, position, direction, energy, input_index, entry_index ]
