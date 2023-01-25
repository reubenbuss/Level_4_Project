import pandas as pd

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_dataset\2 dpi\healthy\1.mspec", sep=",",skiprows=4,names=["Wavelengths","Data"])
s = df["Wavelengths"]
final_df = s.to_frame(name = "Species")
final_df["Day"] = "2"
final_df = final_df[["Day","Species"]]
print(final_df)
#green_control_directory = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Spec_Data\green_control.txt",sep="\t",skiprows=1,names=["folder","file"])
