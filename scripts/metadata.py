#file imports 
import os
import shutil
import json
import logging
import pandas as pd
import sys 


#logging config
logging.basicConfig(filename='..\logs\metadata.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)


#define paths directories
directory=('../data/data/train/wav')
files=[]
target=('../data/alldata')
rootdir = 'path/to/dir'


def merge_files():
    #this function loops through all the folders and extracts all the wav files into one folder 
    for folders in os.listdir(directory):
        #try cacth erros
        try:
        
            print("========== Accessing root directory ============== \n ")
            logging.info("======= Accessing root directory ============= \n")
            
            print("================== Accessing Subfolders ============= \n")
            #accessing subfolders inside train/wav directory
            subfolder = os.path.join(directory,folders)
            
            #looping through all contents in the subfolder
            for wavz in os.listdir(subfolder):

                print("================ Accessing files ============= \n")
                files.append(wavz)
                finalpath = os.path.join(subfolder,wavz)
                #copying files to new audio_path
                print("================ copying files to new audio_path============= \n")
                shutil.copy(finalpath, target)
        except FileNotFoundError:
            print(" !!! Error File Not Found  !!!!! \n")
            print(" !!! Program Failed !!!!! \n")
            logging.error(" !!! Error Program Failed !!!!! \n")
            logging.error("Safely exiting the program")
            print("Safely exiting the program")
            sys.exit(1)
        except Exception as e:
            print(" !!! Error !!!!! \n")
            print (" !!! An excetion occurred Error: {} ".format(e.__class__))
            logging.error(" !!! Error Program Failed !!!!! \n")
            logging.error("Safely exiting the program")
            print("Safely exiting the program")
            sys.exit(1)


name_to_text={}

## creating metadatfile 
def meta_data():
    logging.info("===================== Initializing meat_data function ==================== \n")
    print ("===================== Creating metadata file ================= \n ")
    filename=('../data/data/train/text')
    with open (filename, encoding="utf-8")as f:
        try:
            #open the txt file containnig file and matching text file
            print(" ========= Opening files ========= \n")
            logging.info("===================== Opening text file ========= \n ")
            f.readline()
            for line in f:
                #split first half to be the value and second part is the key/text
                
                name=line.split("\t")[-1]
                name=name.rstrip()
                file=line.split("\t")[0]
                file=file+".wav"
                length=len(name)         
                name_to_text[file]=[name,length]
            print(" ========== Completed =========== \n ")
        except Exception as e:
            print (" !!!! Error !!!! ")
            print (" !!!! The system raised an exception {} !!!!!".format(e.__class__))

    with open("../data/meta_datas.json", "w") as outfile: 
        json.dump(name_to_text, outfile)

def get_file():
    #we convert the json file we got from the previous function into a dataframe
    try:
        print(" ========= Converting dictionary to dataframe ================= \n")
        logging.info("======= Converting dictionary to dataframe ================= \n")

        data = pd.DataFrame.from_dict(name_to_text, orient ='index')
        data=data.reset_index()
        print("======= Creating columns ================= \n")
        data.columns=['wav_file','text','length']
        
        data.to_csv("../data/merged_data.csv",index=False)  
    except Exception as e:
        print (" !!!! Error !!!! ")
        print (" !!!! The system raised an exception {} !!!!!".format(e.__class__))




if (__name__== '__main__'):
    merge_files()
    meta_data()
    get_file()
