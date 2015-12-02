import os
import numpy as np 
import re
import h5py 
from scipy.io import loadmat, savemat
METADATA_TAG = '__metadata'

class jsdict(dict):
    def __init__(self, *args, **kwargs):
        super(jsdict, self).__init__(*args, **kwargs)
        self.__dict__ = self

#straight from micheal hills repo"
def read(filename):
    data = h5py.File(filename, 'r')
    obj = {}
    for key in data.keys():
        value = data[key]
        if key == METADATA_TAG:
            for metakey in value.attrs.keys():
                obj[metakey] = value.attrs[metakey]
        elif not key.startswith('__list'):
            obj[key] = value[:]

    list_keys = [key for key in data.keys() if key.startswith('__list')]
    if len(list_keys) > 0:
        list_keys.sort()
        for key in list_keys:
            match = list_regex.match(key)
            assert match is not None
            list_key = match.group(1)
            list_index = int(match.group(2))
            out_list = obj.setdefault(list_key, [])
            assert len(out_list) == list_index
            out_list.append(data[key][:])

    data.close()

    return jsdict(obj)

pib_spec_ents = [
'0.25-1-1.75-2.5-3.25-4-5-8.5-12-15.5-19.5-24',
'0.25-2-3.5-6-15-24',
'0.25-2-3.5-6-15',
'0.25-2-3.5',
'6-15-24',
'2-3.5-6',
'3.5-6-15'
]

keys = ['corr','freq-corr-1-None','fft_mag_fbin-mean','hfd-2','hurst','pfd']
suffix_fbin = '-0.5-2.25-4-5.5-7-9.5-12-21-30-39-48_log10_fch' 
prefix = 'fft_mag_pib-spec-ent-'

def load_subject_data(subject,test = False):
    subject_dict = {}

    
    for key in keys:
        curkey = key
        if 'fbin' in key:
            curkey = key+suffix_fbin
        if not test:
            preictal_file   = read('./preprocessed/' +subject+'/' + subject + '_preictal_pp_w-75s_' + curkey + '.hdf5')
            interictal_file = read('./preprocessed/'+subject+'/' + subject  +  '_interictal_pp_w-75s_'+ curkey + '.hdf5')
        
            preictal        = preictal_file['X']
            interictal      = interictal_file['X']
    
            subject_dict[key] = (interictal,preictal)
        else:

            test_file = read('./preprocessed/' +subject+'/' + subject + '_test_pp_w-75s_' + curkey + '.hdf5')

            t = test_file['X']
            subject_dict[key] = t
    
    for i in range(0,len(pib_spec_ents)):
        cur_setup = pib_spec_ents[i]

        if not test:
            preictal_file   = read('./preprocessed/' +subject+'/' + subject + '_preictal_pp_w-75s_' + prefix+cur_setup + '.hdf5')
            interictal_file = read('./preprocessed/'+subject+'/' + subject  +  '_interictal_pp_w-75s_'+ prefix+cur_setup+ '.hdf5')
        
            preictal        = preictal_file['X']
            interictal      = interictal_file['X']
    
            subject_dict[prefix+str(i)] = (interictal,preictal)
        else:
            test_file   = read('./preprocessed/' +subject+'/' + subject + '_test_pp_w-75s_' + prefix+cur_setup + '.hdf5')

            t = test_file['X']

            subject_dict[prefix+str(i)] = t
    return subject_dict

def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension

def load_subject_data_cnn(subject,test = False):
    subject_dict = {}
    folder_path = 'preprocessed/cnn/'+subject
    raw_files = get_files_paths(folder_path)
    #print len(raw_files)
    
    
    if test:
        t = []
        for f in raw_files:
            d = loadmat(f)
            x = d['data']
            t.append(x)
        t = np.array(t)
        subject_dict['cnn'] = t
    else:
        preictal = []
        interictal = []
        for f in raw_files:
            d = loadmat(f)
            x = d['data']
            
            if 'interictal' in f:
                #print True
                interictal.append(x)
            elif 'preictal' in f:
                preictal.append(x)
        preictal = np.array(preictal)
        interictal = np.array(interictal)
        subject_dict['cnn'] = (interictal,preictal)
        
    return subject_dict
def load_subjects(subject_list,test = False,cnn = False):
    all_subjects = {}

    for subject in subject_list:
        if not cnn:
            all_subjects[subject] = load_subject_data(subject,test = test)
        else:
            all_subjects[subject] = load_subject_data_cnn(subject,test = test)
    return all_subjects

