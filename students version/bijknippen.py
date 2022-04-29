import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
import tqdm.contrib as tqdmc
import shutil

#######################################################################
show = False #gwn aan of uit zetten, aan is showen uit is bijknippen
place = r"C:\Users\gille\Downloads\hypercubes_rhys"
########################################################################

def read_H5_hypercube(fn, dataset="data"):
    with h5py.File(fn, 'r') as fh:
        if dataset not in fh['hypercube'].keys():  # ammended to read the hypercube format
            return None
        data = np.array(fh['hypercube'][dataset][:])

    if data.dtype == np.uint8:
        pass
    else:
        if dataset !="metadata" and data.max() <= 1:
            data = (data * 255).astype(np.uint8)
    return data

def add_dataset_hypercube(fn, data, dataset="data"):
    with h5py.File(fn, 'r+') as fh:
        if dataset != "metadata":
            fh['hypercube'].create_dataset(dataset, data=data, compression="lzf")
        else:
            fh['hypercube'].create_dataset(dataset, data=data)

    fh.close()

def make_new_directory(fn, names):
    try:
        os.makedirs(place + r'/hypercube_cut')
    except Exception as e:
        print("folder already exists")
    print("Copying to new folder:")
    for i,name in tqdmc.tenumerate(fn):
        hf = h5py.File( place + r'/hypercube_cut/' + names[i], "w")
        hf.create_group("hypercube")
        hf.close()

filenames = glob.glob(place + r'/*h5')
filenames = [fn for fn in filenames if fn.find("cut")==-1]

if show != True:
    names = os.listdir(place)
    make_new_directory(filenames, names) #makes new directory with all the images

    print("\nCropping images:")
    for i, fn in tqdmc.tenumerate(filenames):
        img = read_H5_hypercube(fn, dataset= "data") #load in the image
        img = img[240:2226, 270:1770, :] #cut the image

        metadata = read_H5_hypercube(fn, dataset="metadata")

        fn_new = place + r'/hypercube_cut/' + names[i]
        add_dataset_hypercube(fn_new, img, dataset="data") #this is to save
        add_dataset_hypercube(fn_new, metadata, dataset="metadata")

        old_name = fn_new
        new_name = fn_new.replace("_hypercube", "_cut_hypercube")
        os.rename(old_name, new_name) #to rename the images
    from winsound import Beep
    Beep(550, 50000)
if show == True:
    print("Going to show all images until you type stop")
    filenames = glob.glob(place + r'/*h5')
    i = 0
    while i != len(filenames):
        img = read_H5_hypercube(filenames[i], dataset="data")
        plt.imshow(img[:,:,5])
        plt.show()
        i += 1