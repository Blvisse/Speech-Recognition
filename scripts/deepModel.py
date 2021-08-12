"""

This script train a deep leanring model and returns predictions

"""

import os
import shutil
import json
import logging
import pandas as pd
import sys

logging.basicConfig(filename='..\logs\deepModel.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)


## get the data to begin with

print(" ====== Reading Data ======= ")
try:
    data=pd.read_csv('../data/specData.csv')
    logging.info(" ======= Successfuly loaded data ========= ")
    print("====== Successfully Loaded the data ========  ")
    
except FileNotFoundError:
    
    print(" !!! Error File Not Found  !!!!! \n")
    print(" !!! Program Failed !!!!! \n")
    logging.error(" !!! Error Program Failed !!!!! \n")
    #stop system excution 
    sys.exit(1)

    
except Exception as e:
    
    logging.error(" ==== An error Occured {} ======  ".format(e.__class__))
    print(" ==== An error Occured {} ======  ".format(e.__class__))
    #stop system execution from going on further
    sys.exit(1)
    
    
    
    

##create a function that takes maps the character texts

def char_map(text):
    charlist=[]
    for chars in text:
    
        charlist.append(ord(chars))
        
    return charlist 
        
    
data['char_map']=data['text'].apply(lambda x : char_map(x))



