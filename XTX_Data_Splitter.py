import pandas as pd
from datetime import datetime


fullDF = pd.DataFrame()

def load_de_Monster():
    global fullDF

    startTime = datetime.now()
    fullDF = pd.read_csv('Data//data-training.csv')
    #fullDF = pd.read_csv('Data//SampleXTXData100000.csv')
    print("DF Size")
    print(fullDF.shape)
    print(datetime.now() - startTime)
    print('\n')

def Data_Sample_split(N):
    global fullDF

    startTime = datetime.now()

    sampleDF = fullDF.sample(n = N)
    print("Sample Size")
    print(sampleDF.shape)

    filename = "Data//XTXData{}K.csv".format(str(N/1000))
    sampleDF.to_csv(filename, index=None,header=True)
    print("CSV Created")

    print("--------------------")
    print(datetime.now() - startTime )
    print('\n')


load_de_Monster()
Data_Sample_split(2999999)
