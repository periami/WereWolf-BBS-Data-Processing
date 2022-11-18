from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import japanize_matplotlib
import json
import numpy as np
import pandas as pd
import seaborn as sn
import collections



def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    labels = ["CO", "占い結果", "霊能結果", "護衛先", "CO撤回", "その他"]
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True)
    plt.savefig("./Machine_Learning/Combine_ngram/confusion_matrix.png")
    plt.clf()
    

data_path = "./Machine_Learning/Combine_ngram/combine_ngram.csv"
# label_dic = {"CO":0, "占い結果":1, "霊能結果":2, "護衛先":3, "CO撤回":4, "占い結果撤回":5, "霊能結果撤回":6, "護衛先撤回":7}
label_dic = {"CO":0, "占い結果":1, "霊能結果":2, "護衛先":3, "CO撤回":4, "その他":5}
label_str = []

life_status_list = []
voted_list = []
attacked_list = []

with open("./Data_Processing/Log_Info.json", 'r', encoding="utf-8") as f:
    jsn = json.load(f)
    info = jsn["Information"]
    for dic in info:
        life_status_list.append(list(map(lambda val: val, dic["life status"])))
        voted_list.append(dic["voted"])
        attacked_list.append(dic["attacked"])
        
        if dic["action"] == "占い結果撤回" or dic["action"] == "霊能結果撤回" or dic["action"] == "護衛先撤回":
            label_str.append("その他")
        else:
            label_str.append(dic["action"])

voted_array = np.array(voted_list)[:, np.newaxis]
attacked_array = np.array(attacked_list)[:, np.newaxis]

data = np.loadtxt(data_path, delimiter=',', encoding="utf-8", skiprows=1)
data = np.block([data, np.array(life_status_list)])
data = np.block([data, voted_array])
data = np.block([data, attacked_array])
mm = preprocessing.MinMaxScaler()
data = mm.fit_transform(data)

label_num = 6

label = [label_dic[s] for s in label_str]
label_count = collections.Counter(label)
label_eye = np.eye(label_num)[label]

# print(label_count)


x_train, x_test, y_train, y_test = train_test_split(data, label_eye, test_size=0.2, random_state=0)

# print(x_train.shape, x_test.shape)
input_dim = x_train.shape[1]
print(input_dim)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=input_dim,
                          input_dim=input_dim,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(units=1024),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    # tf.keras.layers.Dense(units=1024),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(units=512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(units=256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(units=6, activation='softmax')
], name='sample_model')

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=256, epochs=30, validation_split=0.2, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("./Machine_Learning/Combine_ngram/acc.png")
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("./Machine_Learning/Combine_ngram/loss.png")
plt.clf()


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print(f"loss:{test_loss}")
print(f"acc:{test_acc}")

predict_classes = model.predict(x_test)
predict = np.argmax(predict_classes, axis=1)
true_classes = np.argmax(y_test, axis=1)

print(confusion_matrix(true_classes, predict))

print_cmx(true_classes, predict)