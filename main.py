from os.path import join, isfile, splitext
from os import listdir
import re

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.backend as K
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Lambda, AveragePooling1D
import pandas as pd
import matplotlib.pyplot as plt


def process_sample(file_name, dir):
    reader = open(join(main_path, 'annotations', dir, file_name), 'r')
    annotation = reader.readlines()[5].split(':')[1].strip()
    annotation_split = annotation.split('{')[1].split('}')[0].split("\"")
    labels = [label for label in annotation_split if label != ' ']
    l = ','.join(labels)

    # load an image from file
    try:
        image = load_img(join(main_path, 'PNGImages', dir, splitext(file_name)[0] + '.png'), target_size=(s, s))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        # prepare the image for the VGG model
        features = feature_extractor.predict(image)
    except FileNotFoundError:
        return None
    return [features[0], l]


def cnn_model():
    # define the input layer
    inp = Input((s, s, 3))
    # load model
    model = VGG16(include_top=False, input_tensor=inp)
    for layer in model.layers:
        layer.trainable = False
    # remove the output layer
    out = model.output
    out = GlobalAveragePooling2D()(out)
    out = Lambda(lambda x: K.expand_dims(x, axis=-1))(out)
    out = AveragePooling1D(2)(out)
    out = Lambda(lambda x: x[:, :, 0])(out)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model


def remove_description(labelset_dict):
    labelset_dict = {k: str.lower(label) for k, label in labelset_dict.items()}
    labelset_dict_red = labelset_dict.copy()
    for id, label in labelset_dict.items():
        new_label = []
        for cat in categories:
            match = re.findall("\B" + cat, label)
            if match:
                new_label.append(cat)
        labelset_dict_red[id] = new_label
    return labelset_dict_red


def tokenize_labels(label_col, dataset):
    labels = set()
    try:
        for row in label_col.values():
            row = row.split(',')
            labels = labels.union(set(row))
    except AttributeError:
        for row in label_col.values():
            labels = labels.union(set(row))

    df_copy = dataset.copy()
    for idx, l in enumerate(labels):
        dataset[l] = 0
        class_ref[l] = idx

    for index, row in df_copy.iterrows():
        labels = label_col[index]
        for l in labels:
            dataset.at[index, l] = 1
    return dataset


DATA_DIR = "D:\Datasets"
DATA_HEADER = "mit-csail"
main_path = join(DATA_DIR, DATA_HEADER)

# categories = ['bicycle', 'bus', 'cat', 'car', 'cow', 'dog', 'horse', 'motorbike', 'person', 'sheep']
categories = (['screen', 'keyboard', 'mouse', 'mousepad', 'speaker', 'trash', 'poster', 'bottle',
              'chair', 'can', 'mug', 'light', 'apple', 'car', 'trafficlight', 'bicycle', 'bookshelf', 'building',
              'cd', 'coffeemachine', 'cpu', 'desk', 'door', 'donotenter', 'filecabinet', 'firehydrant',
              'freezer', 'head', 'mug', 'oneway', 'papercup', 'parkingmeter', 'person', 'pot', 'printer',
              'roadregion', 'shelves', 'sky', 'sofa', 'steps', 'stop', 'street', 'streetsign', 'tablelamp',
              'torso', 'astree', 'walkside', 'clock', 'watercooler', 'window', 'streetlight'])

s = 224
features = {}
labelset_dict = {}
class_ref = {}

feature_extractor = cnn_model()
dirs = listdir(join(main_path, 'annotations'))
for dir in dirs:
    for file in listdir(join(main_path, 'annotations', dir)):
        if file.endswith('.txt') and isfile(join(main_path, 'PNGImages', dir, splitext(file)[0]+'.png')):
            features[splitext(file)[0]], labelset_dict[splitext(file)[0]] = process_sample(file, dir)

samples = pd.DataFrame(data=features.values(), index=features.keys())
labelset_dict = remove_description(labelset_dict)
samples['labels'] = labelset_dict.values()
samples = tokenize_labels(labelset_dict, samples)
label_counts = {k:samples[k].sum() for k, _ in class_ref.items()}

samples.to_csv(join(main_path, 'mit-csail.csv'))
writer = open(join(main_path, 'mit-csail-dict.txt'), 'w')
[writer.write(key+':'+str(value)+'\n') for key, value in class_ref.items()]
writer.close()

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xticklabels(label_counts.keys())
ax.bar(label_counts.keys(), label_counts.values())
plt.savefig(join(main_path, 'mit-csail.png'))
plt.close()

