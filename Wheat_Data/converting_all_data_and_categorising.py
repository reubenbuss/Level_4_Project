import pandas as pd
from glob import glob as glob
import numpy as np

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_dataset\2 dpi\healthy\1.mspec", sep=",",skiprows=4,names=["Wavelengths","Data"])
s = df["Wavelengths"]
s = s.tolist()
s = ["Day","Condition"]+s
df_final = pd.DataFrame(s)
df_final = df_final.T
print(df_final)
#green_control_directory = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\green_control.txt",sep="\t",skiprows=1,names=["folder","file"])


for i in glob(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_dataset\*"):#i = days folder
    for j in glob(i+"\*"): #j = infected or healthy folder
        calibration_list = np.zeros(3648).astype(float)
        main_list=[]
        for k in glob(j+"\*cal*"):
            df = pd.read_csv(k,sep=",",skiprows=4,names=["wavelength","data"])
            calibration_list += (df["data"].to_numpy()).astype(float)
        calibration_list = calibration_list/len(glob(j+"\*cal*"))
        for k in glob(j+"\[!cal]*"):
            df =  pd.read_csv(k,sep=",",skiprows=4,names=["wavelength","data"])
            main_list.append(list(df["data"].to_numpy()/calibration_list.astype(float)))
        df = pd.DataFrame(main_list)
        length = len(glob(j+"\[!cal]*"))
        day_list = [i.split("\\")[-1]]*length
        condition_list = [j.split("\\")[-1]]*length
        df1 = pd.concat([pd.DataFrame(day_list),pd.DataFrame(condition_list),df],axis=1,ignore_index=True)
        df_final = pd.concat([df_final,df1],ignore_index=True)

(df_final).to_csv("Wheat_data.csv",index = False)
print("Finished")
