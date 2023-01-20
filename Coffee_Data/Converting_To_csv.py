import pandas as pd
import glob

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Shaw_valley\Avo_1_Reflection_01-31-23-108.txt", sep="\t",skiprows=13,names=["Wavelengths","Data"])
s = df["Wavelengths"]
final_df = s.to_frame(name = "Wavelengths")
# All files and directories ending with .txt and that don't begin with a dot:
files = glob.glob(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Shaw_valley\*.txt")
for i in files:
    df = pd.read_csv(i,sep="\t",skiprows=13,names=["Wavelength","data"])
    final_df[i[103:-4]] = df["data"]


final_df = final_df.set_index('Wavelengths').T
print(final_df.head())
final_df.to_csv("Coffee_data.csv")
#final_df.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Coffee_Data.csv",index=False)
print("Finished")
