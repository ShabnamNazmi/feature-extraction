from os.path import join, isfile, splitext
from os import listdir

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.backend as K
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Lambda, AveragePooling1D

import pandas as pd


def read_sample(file_name, dir):
    reader = open(join(main_path, 'annotations',  file_name), 'r')
    annotation = reader.readlines()[5].split(':')[1].strip()
    annotation_split = annotation.split('{')[1].split('}')[0].split("\"")
    labels = [label for label in annotation_split if label != ' ']
    l = ','.join(labels)

    # load an image from file
    try:
        image = load_img(join(main_path, 'PNGImages',  splitext(file_name)[0] + '.png'), target_size=(s, s))
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


def process_labels(label_col, samples):
    labels = set()
    for row in label_col.values():
        row = row.split(',')
        labels = labels.union(set(row))

    df_copy = samples.copy()
    for idx, l in enumerate(labels):
        samples[l] = 0
        class_ref[l] = idx

    for index, row in df_copy.iterrows():
        labels = row['labels'].split(',')
        for l in labels:
            if l != '' and l != ', ' and l != ',':
                samples.at[index, l] = 1


DATA_DIR = "D:\Datasets"
DATA_HEADER = "pascal-voc6"
main_path = join(DATA_DIR, DATA_HEADER)

s = 224
features = {}
label_col = {}
class_ref = {}

feature_extractor = cnn_model()
dirs = listdir(join(main_path, 'annotations'))
dir = None
# for dir in dirs:
for file in listdir(join(main_path, 'annotations')):
    if file.endswith('.txt') and isfile(join(main_path, 'PNGImages', splitext(file)[0]+'.png')):
        features[splitext(file)[0]], label_col[splitext(file)[0]] = read_sample(file, dir)

samples = pd.DataFrame(data=features.values(), index=features.keys())
samples['labels'] = label_col.values()
process_labels(label_col, samples)


def remove_labels(self, data):
    class_dict = dict(zip(list(data.columns)[NO_FEATURES:-1], [0] * self.label_count))
    for tag in class_dict.keys():
        count = data[tag].sum()
        class_dict[tag] = count
    trunc_tags = [tag for tag in class_dict.keys() if class_dict[tag] >= 100]
    self.class_dict_trunc = {key: val for key, val in class_dict.items() if key in trunc_tags}

    data_copy = data.copy()
    drop_tags = self.class_dict_trunc.keys()
    for labelset in data['labelset']:
        new_labelset = set([l for l in labelset if self.class_dict_trunc.get()])
        print(3)
    return data

# samples.drop('labels', axis=1, inplace=True)
samples.to_csv(join(main_path, 'voc6.csv'))
