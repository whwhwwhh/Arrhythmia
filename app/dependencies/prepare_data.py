import pandas as pd
import os
from sklearn.preprocessing import normalize

    
#this function is specaily designed for the arraythmia dataset.
def load_arraythmia_data(path:str):
    try:
        df = pd.read_csv(path)
        print("INFO: the data is loaded.")
        #according to my analysis (in jupyter notebook), I decided to remove following columns
        df.drop(columns = [10, 11, 12, 13,14], inplace = True)
        
        #next step, build labels. I wanted to convert the task to be a binary classification.
        binary_labels = []
        for label in df[279]:
            if label == 1:
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        #now we need to drop the original labels
        df.drop(columns=[279], inplace = True)
        #convert dataframe to numpy array
        
        return df.to_numpy(), binary_labels
    except Exception as e:
        print("ERROR:the input file should be in csv format!")
        return None, None