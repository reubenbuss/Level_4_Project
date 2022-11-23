import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
tf.random.set_seed(42)

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
data = pd.read_csv(file)
data.drop(["SpectraID","WhiteReference","ContactProbe",
    "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
    "Location","Lat","Long","StandAge","StandHealth",
    "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species
lb = LabelEncoder()
y = lb.fit_transform(y)
#x=data.drop(["Species"],axis=1,inplace=False)
x=data[["350","356","713","351","1890","1888","1891","375","1893","2399","1899","355","1651","360","723","2241","1675","717","352","729","368","1998","1892","1831","353","1894","728","512","1882","1878","1123","2014","690","664","661","366","2132","1901","1886","1648"]]
#x=data[["350","1901"]]

x=x.values.reshape(x.shape[0],x.shape[1],1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)


model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(16,2,activation="relu",input_shape=(40,1)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Flatten(),
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
model.fit(x_train, y_train, epochs=cycles, verbose=1,batch_size=16)

acc = model.evaluate(x_test, y_test)
print("Loss:", acc[0], " Accuracy:",acc[1])

pred=model.predict(x_test)
y_pred = np.argmax(pred,axis=1)
y_test = np.argmax(y_test,axis=1)
cm=confusion_matrix(y_test,y_pred)
print(cm)
cmd_obj = ConfusionMatrixDisplay(cm)
cmd_obj.ax_.set(title="Sklearn Confusion Matrix",xlabel="Predicted Species",ylabel="Actual Species")
plt.show()
