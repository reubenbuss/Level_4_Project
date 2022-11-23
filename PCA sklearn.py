import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
tf.random.set_seed(42)

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
data = pd.read_csv(file)
data.drop(["SpectraID","WhiteReference","ContactProbe",
    "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
    "Location","Lat","Long","StandAge","StandHealth",
    "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species
x=data.drop(["Species"],axis=1,inplace=False)

#PCA
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=4) #96%
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)


#DNN
lb = LabelEncoder()
y = lb.fit_transform(y)
x=principalDf

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

cycles = 40
history = model.fit(x_train, y_train, validation_data = (x_test,y_test), epochs=cycles, verbose=0,batch_size=16)


rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

plt.plot(
    np.arange(1, cycles+1), 
    history.history['loss'], label='Loss',color = "navy"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['accuracy'], label='Accuracy', color = "deepskyblue"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['precision'], label='Precision', color = "dodgerblue"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['recall'], label='Recall', color = "royalblue"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['val_loss'], label='Val_Loss', color = "firebrick"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['val_accuracy'], label='Val_Accuracy', color = "orangered"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['val_precision'], label='Val_Precision', color = "tomato"
)
plt.plot(
    np.arange(1, cycles+1), 
    history.history['val_recall'], label='Val_Recall', color = "salmon"
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.ylim(0,1)
plt.show()



