
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv"
data = pd.read_csv(file)
data = data[["Species","1000"]]

red_wavelengths = data[data["Species"] == "Red"]["1000"]
white_wavelengths = data[data["Species"] == "White"]["1000"]
black_wavelengths = data[data["Species"] == "Black"]["1000"]
na_wavelengths = data[data["Species"] == "na"]["1000"]
mud_wavelengths = data[data["Species"] == "Mud"]["1000"]

fig, ax = plt.subplots(figsize = (12,7))
ax.set_title("Boxplots to show spread of data at 1000nm")
dataset = [red_wavelengths,white_wavelengths,black_wavelengths,mud_wavelengths,na_wavelengths]
labels = ["Red","White","Black","Mud","Na"]
ax.boxplot(dataset,labels = labels)
plt.show()




