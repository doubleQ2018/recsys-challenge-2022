# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import os

block_num = 6

def downcast(series,accuracy_loss = True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()
        
        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif  max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series
        
    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)
        
        if accuracy_loss:
            max_value = series.max()
            min_value = series.min()
            if np.isnan(max_value):
                max_value = 0
            
            if np.isnan(min_value):
                min_value = 0
                
            if min_float_type=='float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if(len(tmp)==0):
                return series.astype(np.float16)
            
            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)
           
            else:
                return series
            
    elif series.dtype == np.object:
        try:
            return series.astype(np.float16)
        except:
            return series
    else:
        return series

def load_pickle(file_path):
    file_dir = file_path[:-4]
    if ('feat_data' in file_path) and os.path.exists(file_dir):
        datas = []
        for block_id in range(block_num):
            file_path = file_dir+'/'+str(block_id)+'.pkl'
            datas.append( pickle.load( open(file_path,'rb') ) )
        data = pd.concat( datas )
        return data
    else:
        return pickle.load( open(file_path,'rb') )
    
    
def dump_pickle(obj, file_path):
    if ('feat_data' in file_path) and (type(obj) == pd.core.frame.DataFrame) and obj.shape[1]>=5 :
        block_len = len(obj)//block_num
        file_dir = file_path[:-4]
        if not os.path.exists(file_dir):
            os.makedirs( file_dir )
        for block_id in range(block_num):
            file_path = file_dir+'/'+str(block_id)+'.pkl'
            l = block_id * block_len
            r = (block_id+1) * block_len
            if block_id == block_num - 1:
                pickle.dump( obj.iloc[l:], open(file_path,'wb') )
            else:
                pickle.dump( obj.iloc[l:r], open(file_path,'wb') )
    else:
        pickle.dump(obj, open(file_path,'wb'))
    

