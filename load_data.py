import os
import numpy as np 
import re
import h5py 

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

def load_subject_data(subject):
    subject_dict = {}

    suffix_fbin = '-0.5-2.25-4-5.5-7-9.5-12-21-30-39-48_log10_fch' 
    for key in keys:
        curkey = key
        if 'fbin' in key:
            curkey = key+suffix_fbin
        preictal_file   = read('./preprocessed/' +subject+'/' + subject + '_preictal_pp_w-75s_' + curkey + '.hdf5')
        interictal_file = read('./preprocessed/'+subject+'/' + subject  +  '_interictal_pp_w-75s_'+ curkey + '.hdf5')
        
        preictal        = preictal_file['X']
        interictal      = interictal_file['X']
    
        subject_dict[key] = (interictal,preictal)

    prefix = 'fft_mag_pib-spec-ent-'
    for i in range(0,len(pib_spec_ents)):
        cur_setup = pib_spec_ents[i]

        preictal_file   = read('./preprocessed/' +subject+'/' + subject + '_preictal_pp_w-75s_' + prefix+cur_setup + '.hdf5')
        interictal_file = read('./preprocessed/'+subject+'/' + subject  +  '_interictal_pp_w-75s_'+ prefix+cur_setup+ '.hdf5')
        
        preictal        = preictal_file['X']
        interictal      = interictal_file['X']
    
        subject_dict[prefix+str(i)] = (interictal,preictal)
    
    return subject_dict

def load_subjects(subject_list):
    all_subjects = {}

    for subject in subject_list:
        all_subjects[subject] = load_subject_data(subject)

    return all_subjects