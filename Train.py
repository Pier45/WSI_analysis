from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class CreationTFRecord:
    def __init__(self):
        pass


class Training:
    def __init__(self):
        pass

def load_image(addr, i):
    # read an image and resize to (nx, ny)
    # cv2 load images as BGR, convert it to RGB
    #               ??????
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('next image {}'.format(i))
    return img


def createDataRecord(out_filename, addrs, labels):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        img = load_image(addrs[i], i)
        label = labels[i]

        if img is None:
            continue
        # Create a feature IMPORTANTE!!!!
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def counter(sel_path):
    list_files = glob.glob(sel_path)
    n_elements = len(list_files)
    return list_files, n_elements


def select_folder(path):
    ind1, ind2, ind3, ind4, ind5 = [], [], [], [], []
    tot_ele = [0] * 6
    folders = os.listdir(path)
    for i in folders:
        sel_path = path + '/' + str(i) + '/*.png'
        if i[-1] == 'C':
            cl = 1
            list_files, n_elements = counter(sel_path)
            print(f'cartella:  {i}   -- numero elementi: {n_elements}')
            tot_ele[cl] = tot_ele[cl] + n_elements
            ind1.extend(list_files[:])
        elif i[-1] == 'H':
            cl = 2
            list_files, n_elements = counter(sel_path)
            print(f'cartella:  {i}   -- numero elementi: {n_elements}')
            tot_ele[cl] = tot_ele[cl] + n_elements
            ind2.extend(list_files[:])
        elif i[-1] == 'r':
            cl = 3
            list_files, n_elements = counter(sel_path)
            print(f'cartella:  {i}   -- numero elementi: {n_elements}')
            tot_ele[cl] = tot_ele[cl] + n_elements
            ind3.extend(list_files[:])
        elif i[-1] == 'T':
            cl = 4
            list_files, n_elements = counter(sel_path)
            print(f'cartella:  {i}   -- numero elementi: {n_elements}')
            tot_ele[cl] = tot_ele[cl] + n_elements
            ind4.extend(list_files[:])
        elif i[-1] == 'V':
            cl = 5
            list_files, n_elements = counter(sel_path)
            print(f'cartella:  {i}   -- numero elementi: {n_elements}')
            tot_ele[cl] = tot_ele[cl] + n_elements
            ind5.extend(list_files[:])
        else:
            pass

    cl_min = tot_ele[1:6].index(min(tot_ele[1:6])) + 1
    mini = min(tot_ele[1:6])
    print(tot_ele[1:6], cl_min, mini)
    dataset = []
    lab = []
    dataset = random.choices(ind1, k=mini) + random.choices(ind2, k=mini) + random.choices(ind3,
                                                                                           k=mini) + random.choices(
        ind4, k=mini) + random.choices(ind5, k=mini)
    lab = [1] * mini + [2] * mini + [3] * mini + [4] * mini + [5] * mini

    print(f' DATASET: {len(dataset)}    --  LAB: {len(lab)}')

    return dataset, lab


addrs, labels = select_folder('/content/drive/My Drive/CRC_reduced_bioinf')

# to shuffle data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

# Divide the data into 70% train, 10% validation, and 20% test
train_addrs = addrs[0:int(0.7 * len(addrs))]
train_labels = labels[0:int(0.7 * len(labels))]
val_addrs = addrs[int(0.7 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.7 * len(addrs)):int(0.8 * len(addrs))]
test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]

createDataRecord('train.tfrecords', train_addrs, train_labels)
createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('test.tfrecords', test_addrs, test_labels)
