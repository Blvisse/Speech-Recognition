import dvc.api
import pandas as pd
data=dvc.api.get_url('data/testData.csv','https://github.com/Blvisse/Speech-Recognition',rev='gtest-v0')
# import os


data=pd.read_csv(data)
# arr = os.listdir(data)
# print(arr)


##comment section

print (data)