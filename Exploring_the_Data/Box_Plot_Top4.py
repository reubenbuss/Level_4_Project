import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
data = pd.read_csv(file)
wavelengths = ["750","680","351","390"]
wavelengths.append("Species")
data = data[wavelengths]
wavelengths.remove("Species")

fig = plt.figure(dpi = 2400)
ax = plt.axes()

def split_into_species(data):
    red_wavelengths = data[data["Species"] == "Red"]
    white_wavelengths = data[data["Species"] == "White"]
    black_wavelengths = data[data["Species"] == "Black"]
    return red_wavelengths,white_wavelengths,black_wavelengths

def split_into_wavelengths(data,wavelengths):
    red_wavelengths,white_wavelengths,black_wavelengths = split_into_species(data)
    new = []
    for i in wavelengths:
        a = red_wavelengths[i].to_list()
        b = white_wavelengths[i].to_list()
        c = black_wavelengths[i].to_list()
        new.append([a,b,c])
    return new

def wavelength_combiner(data,wavelengths):
    new = split_into_wavelengths(data,wavelengths)
    res1 = new[0]
    res2 = new[1]
    res3 = new[2]
    res4 = new[3]
    return res1,res2,res3,res4

def create_plot(data,wavelengths):
    a,b,c,d = wavelength_combiner(data,wavelengths)
    plt.boxplot(a, positions = [1,2,3], widths = 0.6)
    plt.boxplot(b, positions = [5,6,7], widths = 0.6)
    plt.boxplot(c, positions = [9,10,11], widths = 0.6)
    plt.boxplot(d, positions = [13,14,15], widths = 0.6)


create_plot(data,wavelengths)
plt.show()
