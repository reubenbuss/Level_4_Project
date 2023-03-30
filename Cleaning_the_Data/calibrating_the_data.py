import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\data_to_be_calibrated.csv")


header = df.columns.to_numpy()
for i in range(0,df.shape[0],7):
    
    cal = (df.iloc[i+5,1:].to_numpy()+df.iloc[i+6,1:].to_numpy())/2
    value1 = (df.iloc[i,1:].to_numpy()/cal)
    value2 = (df.iloc[i+1,1:].to_numpy()/cal)
    value3 = (df.iloc[i+2,1:].to_numpy()/cal)
    value4 = (df.iloc[i+3,1:].to_numpy()/cal)
    value5 = (df.iloc[i+4,1:].to_numpy()/cal)
    value1 = np.insert(value1,0,df.iloc[i,0])
    value2 = np.insert(value2,0,df.iloc[i,0])
    value3 = np.insert(value3,0,df.iloc[i,0])
    value4 = np.insert(value4,0,df.iloc[i,0])
    value5 = np.insert(value5,0,df.iloc[i,0])
    header = np.vstack((header,value1,value2,value3,value4,value5))

new_df = pd.DataFrame(header)


new_df.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Calibrated_data.csv", index=False)

print("Finished")
