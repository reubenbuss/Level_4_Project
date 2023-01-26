import pandas as pd
from glob import glob as glob
import numpy as np

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_dataset\2 dpi\healthy\1.mspec", sep=",",skiprows=4,names=["Wavelengths","Data"])
s = df["Wavelengths"]
s = s.tolist()
s = ["Day","Condition"]+s
df_final = pd.DataFrame(s)
df_final = df_final.T #dataframe to act as wavelength header

for i in glob(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_dataset\*"):#i = days folder
    for j in glob(i+"\*"): #j = infected or healthy folder
        calibration_list = np.zeros(3648).astype(float)
        main_list=[]
        for k in glob(j+"\*cal*"): #calibration files
            df = pd.read_csv(k,sep=",",skiprows=4,names=["wavelength","data"])
            calibration_list += (df["data"].to_numpy()).astype(float) #sum of all calibration data as a vector
        calibration_list = calibration_list/len(glob(j+"\*cal*")) #dividing sum of calibration data by number of calibration files to get average 
        for k in glob(j+"\[!cal]*"): #non-calibration files aka data files
            df =  pd.read_csv(k,sep=",",skiprows=4,names=["wavelength","data"]) #open data files
            main_list.append(list(df["data"].to_numpy()/calibration_list.astype(float))) #data vector divide by calibration vector element-wise
        df = pd.DataFrame(main_list) #make into dataframe
        length = len(glob(j+"\[!cal]*")) #number of data files
        day_list = [i.split("\\")[-1]]*length #day index
        condition_list = [j.split("\\")[-1]]*length #condition index
        df1 = pd.concat([pd.DataFrame(day_list),pd.DataFrame(condition_list),df],axis=1,ignore_index=True) #combining columns of day and conditon index and data to create a day dataframe
        df_final = pd.concat([df_final,df1],ignore_index=True) #adding each day dataframe together with wavelengths at the top

(df_final).to_csv("Wheat_data.csv",index = False)
print("Finished")
