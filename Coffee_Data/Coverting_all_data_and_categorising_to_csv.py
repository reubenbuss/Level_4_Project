import pandas as pd
import glob

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Shaw_valley\Avo_1_Reflection_01-31-23-108.txt", sep="\t",skiprows=13,names=["Wavelengths","Data"])
s = df["Wavelengths"]
final_df = s.to_frame(name = "Wavelengths")
# All files and directories ending with .txt and that don't begin with a dot:
files1 = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\green_control.txt",sep="\t",skiprows=1,names=["folder","file"])
files2 = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Rust.txt",sep="\t",skiprows=1,names=["folder","file"])
files3 = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Rust_Canopy.txt",sep="\t",skiprows=1,names=["folder","file"])
files4 = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\AribicavarGeisha.txt",sep="\t",skiprows=1,names=["folder","file"])

p = r'C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}'.format(folder = files1.iloc[0,0],file = files1.iloc[0,1])
print(p)

for i in range(0,files1.shape[0]):
    df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}".format(folder = str(files1.iloc[i,0]),file=str(files1.iloc[i,1])),sep="\t",skiprows=13,names=["Wavelength","data"])
    final_df["green_control"] = df["data"]


#final_df = final_df.set_index('Wavelengths').T
print(final_df.head())
#final_df.to_csv("Coffee_data_classified.csv")
#final_df.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Coffee_Data.csv",index=False)
print("Finished")
