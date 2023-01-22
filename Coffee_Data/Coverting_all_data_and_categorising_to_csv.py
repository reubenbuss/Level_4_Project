import pandas as pd
import glob

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Shaw_valley\Avo_1_Reflection_01-31-23-108.txt", sep="\t",skiprows=13,names=["Wavelengths","Data"])
s = df["Wavelengths"]
final_df = s.to_frame(name = "Species")
# All files and directories ending with .txt and that don't begin with a dot:
green_control_directory = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\green_control.txt",sep="\t",skiprows=1,names=["folder","file"])
Rust_directory = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Rust.txt",sep="\t",skiprows=1,names=["folder","file"])
Rust_Canopy_directory = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\Rust_Canopy.txt",sep="\t",skiprows=1,names=["folder","file"])
AribicavarGeisha_directory = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\AribicavarGeisha.txt",sep="\t",skiprows=1,names=["folder","file"])

p = r'C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}'.format(folder = green_control_directory.iloc[0,0],file = green_control_directory.iloc[0,1])
print(p)

for i in range(0,green_control_directory.shape[0]):
    df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}".format(folder = str(green_control_directory.iloc[i,0]),file=str(green_control_directory.iloc[i,1])),sep="\t",skiprows=13,names=["Wavelength","green_control"])
    final_df = pd.concat([final_df.reset_index(drop=True,inplace=False),df["green_control"].reset_index(drop=True,inplace=False)],axis=1)

for i in range(0,Rust_directory.shape[0]):
    df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}".format(folder = str(Rust_directory.iloc[i,0]),file=str(Rust_directory.iloc[i,1])),sep="\t",skiprows=13,names=["Wavelength","Rust"])
    final_df = pd.concat([final_df.reset_index(drop=True,inplace=False),df["Rust"].reset_index(drop=True,inplace=False)],axis=1)

for i in range(0,Rust_Canopy_directory.shape[0]):
    df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}".format(folder = str(Rust_Canopy_directory.iloc[i,0]),file=str(Rust_Canopy_directory.iloc[i,1])),sep="\t",skiprows=13,names=["Wavelength","Rust_Canopy"])
    final_df = pd.concat([final_df.reset_index(drop=True,inplace=False),df["Rust_Canopy"].reset_index(drop=True,inplace=False)],axis=1)

for i in range(0,AribicavarGeisha_directory.shape[0]):
    df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\{folder}\{file}".format(folder = str(AribicavarGeisha_directory.iloc[i,0]),file=str(AribicavarGeisha_directory.iloc[i,1])),sep="\t",skiprows=13,names=["Wavelength","AribicavarGeisha"])
    final_df = pd.concat([final_df.reset_index(drop=True,inplace=False),df["AribicavarGeisha"].reset_index(drop=True,inplace=False)],axis=1)


final_df = final_df.set_index('Species').T
print(final_df.head())
#final_df.to_csv("Coffee_data_classified.csv")
final_df.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data.csv")
print("Finished")
