"""turn off sciview in File -> Settings.... -> Tools -> SciView"""


"""this annotator version is originally created by Edgar Cardenas. Updated by Ritchie Heirmans. Further developed by bachelor students, with some additional improvements
added functionality:
 - multi label
 - improved GUI
 - Folder selection through GUI
 - Label info from excel sheet:  column header name should be 'label'
 - label_legend created in export h5 file
 
 a mask image is created with 1 band. the pixel value is dependant on the layer number: 1,2,3,...
 
  
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon2mask, disk
from skimage import io
from copy import deepcopy
import cv2
import easygui as eg
import pandas as pd
import sys

#############################################################################

class menu():
    def __init__(self,exclude_labeled=True):
        self.folder = ""
        self.excell = None
        self.labels = []
        self.exclude_labeled = exclude_labeled
        self.mainmenu()

    def mainmenu(self):
        event = eg.indexbox("Choose user","yoinky :>",["Choose save path","Choose excell path", f"exclude labeled: {self.exclude_labeled}", "klaar"])
        if event == 0:
            self.pathmenu()
        elif event == 1:
            self.excellmenu()
        elif event == 2:
            self.exclude_labeled = not self.exclude_labeled
            self.mainmenu()
        elif event == 3:
            pass
        else:
            print('exit menu')
        
    def excellmenu(self):
        try:
            if self.folder != None:
                os.chdir(self.folder)
            self.excell = eg.fileopenbox("Choose excelldocumentje", default="*.xlsx", filetypes=["*.xls", "*.xlsx"])
            df = pd.read_excel (self.excell)
            mylist = df['label'].tolist()
            for i in range(len(mylist)):
                self.labels.append([mylist[i]])
        except:
            print('no labels imported')
        #if self.folder != "" and self.excell != None:
        #    pass
        self.mainmenu()
 

    def pathmenu(self):
        self.folder = eg.diropenbox("Choose place to save")
        if self.folder != "" and self.excell == None:
            self.excellmenu()
        elif self.folder != "" and self.excell != None:
            pass
        else:
            self.mainmenu()
#############################################################################



###############################
#  Annotator 
###############################

class Annotate():

    def __init__(self, exclude_labeled=True, run=True, bands=[1]):

        self.m = menu(exclude_labeled)
        exclude_labeled = self.m.exclude_labeled
        self.excell = self.m.excell
        if not os.path.exists(self.m.folder):
            print('folder does not exist')
            sys.exit()
        searchpath = self.m.folder + r'/*h5'
        print(searchpath)
        filenames = glob.glob(searchpath)

        if exclude_labeled:
            filenames = [fn for fn in filenames if not os.path.exists(fn.replace("_hypercube", "_labeled_hypercube"))]
            filenames = [fn for fn in filenames if fn.find("_labeled_hypercube") == -1]
        print(filenames)
        if len(filenames) == 0:
            print("no images")
            return

        self.layers = bands
        self.band = 0

        self.filenames = filenames
        self.im_idx = 0
        self.cur_idx = 0
        self.cmap = "gray"
        self.fill = 0.25
        self.clrs_lines = ["#fc0303","#fc8003","#fcce03","#e8f2a7","#3dfc03","#ffffff","#03f4fc","#0384fc","#ff42bd","#000000"]
        self.clrs_fill = ["#ffffff","#fc0303","#cafc03","#1403fc"]
        self.drawn = []

        self.model = None
        self.draw_prediction = False
        self.draw_label = False

        self.polygons = [[] for i in range(len(filenames))]
        self.cur_polygons = []

        self.nbands = None

        if run:
            self.open_window()

    def open_window(self):

        plt.rcParams['keymap.fullscreen'] = []
        fig, ax = plt.subplots(figsize=(15, 12))
        self.fig = fig
        self.ax = ax
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        self.draw_image()

    def read_image(self, filename):

        im = None
        if ".jpg" in filename.lower() or ".tif" in filename.lower():
            im = cv2.imread(filename)
        if "hypercube.h5" in filename.lower():
            im = read_H5_hypercube(filename, all=self.layers)
            print(im.shape)
        elif ".h5" in filename.lower():
            im = read_H5(filename)

        return im

    def load_labeled(self):

        filename = self.filenames[self.im_idx]

        im = None
        if ".jpg" in filename.lower() or ".tif" in filename.lower():
            im = cv2.imread(filename)
        if "hypercube.h5" in filename.lower():
            im = read_H5_hypercube(filename, dataset="labels")
        elif ".h5" in filename.lower():
            im = read_H5(filename, dataset="labels")
        if im is not None:
            if im.dtype == np.bool:        im = im * 255
            if im.ndim == 2:             im = im[..., None]
            im = im.astype(np.uint8)

        if im.ndim == 3:
            self.bands = im.shape[2]
        else:
            self.bands = 1

        return im

    def load_data(self):

        filename = self.filenames[self.im_idx]

        im = self.read_image(filename)

        if im is None:
            print(str(self.im_idx) + ": Not Found")
            self.filenames.pop(self.im_idx)
            self.polygons.pop(self.im_idx)

            im = self.load_data()

        if im.dtype == np.bool_:        im = im * 255
        if im.ndim == 2:             im = im[..., None]
        im = im.astype(np.uint8)
        if im.ndim == 3:
            self.bands = im.shape[2]
        else:
            self.bands = 1
        return im

    def draw_image(self):

        hc = self.load_data()
        im = hc[:, :, self.band]  # choose band in case of hyperspectral data, band is a list, THIS IS THE ONE YOU DRAW LATER ON
        self.ax.clear()

        if self.model is not None:
            if self.draw_prediction:
                im_in = np.mean(im, axis=2)
                pred = self.model.segment(im_in)

                im = np.mean(im, axis=2)
                im = im[..., None] * [1, 1, 1]

                pred = pred.transpose([1, 2, 0])
                # pred = pred > 0.8
                im = channels2rgb(pred)

        if self.draw_label:

            labeled = self.load_labeled()

            if labeled is not None:
                im = channels2rgb(labeled)
        if im.shape[-1] == 1:
            im = im * [1, 1, 1]
        im = im.astype(np.uint8)
        self.ax.imshow(im)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        #tables
        instr = [["←: Last Image"], ["→: Next Image"],
                 ["↑: Next label"], ["↓: Previous label"],
                 ["Backspace: Remove Point"], ["Enter/Right Click: Next Object"],
                 ["Escape: Close Tool"], ["f: Toggle fill opacity " + "(" + str(self.fill) + ")"], ["i: More information"]]
        table_instr = plt.table(cellText=instr, loc='right', cellLoc='left')
        table_instr.set_fontsize(12)
        table_instr.scale(0.5, 2.5)

        fn_str = self.filenames[self.im_idx]
        fn_str = ("..." + fn_str[-20:] if len(fn_str) > 20 else fn_str)
        currentlabeltxt = [['◄',"Label " + str(self.cur_idx+1), '►',fn_str]]
        table_curlabel = plt.table(cellText=currentlabeltxt, loc='top', cellLoc='center')
        table_curlabel.set_fontsize(16)
        table_curlabel.scale(1, 3)

        #colors = plt.cm.BuPu(np.linspace(0, 0.5, 5))
        colors = self.clrs_lines
        cell_text = []
        for i,j in enumerate(self.m.labels):
            cell_text.append([f'label {i+1}'])
        table_legend = plt.table(cellText=cell_text,rowLabels=self.m.labels, loc='left', cellLoc='left', rowColours=colors)
        table_legend.set_fontsize(10)
        table_legend.scale(0.2, 2)
        plt.subplots_adjust(left=0.2, bottom=0.2)

    ###########################################
    ##  UI
    ###########################################
    def onKey(self, event):
        if event.key == 'right':

            filenames = self.filenames
            self.polygons[self.im_idx] = self.cur_polygons
            poly = self.polygons[self.im_idx]
            self.change_image(1)
            self.draw_image
            self.update()

        if event.key == 'left':
            self.change_image(-1)
            self.draw_polygons() #bij left right up down self.draw_polygons() toegevoegd zodat de tables meteen updaten ipv eerst te moeten klikken
            self.draw_image
            self.update()

        if event.key == 'up':
            self.cur_idx += 1
            self.draw_image()
            self.submit_polygon()
            self.draw_polygons()
            

        if event.key == 'down':
            self.cur_idx = max(0, self.cur_idx - 1)
            self.submit_polygon()
            self.draw_image()
            self.update()

        if event.key == 'enter':
            self.submit_polygon()

        if event.key == 'escape':
            self.polygons[self.im_idx] = self.cur_polygons
            plt.close(self.fig)

        if event.key == 'backspace':
            if len(self.cur_polygons[-1]["pts"]) > 0:
                self.cur_polygons[-1]["pts"].pop()
            elif len(self.cur_polygons) > 1:
                self.cur_polygons.pop()
            self.draw_polygons()

        if event.key == "f":
            self.fill = (self.fill + 0.25) % 1
            self.draw_image()
            self.draw_polygons()
            self.update()

        if event.key == "j":
            self.cmap = "jet"
            self.draw_image()

        if event.key == "g":
            self.cmap = "gray"
            self.draw_image()

        if event.key == "l":
            self.draw_label = not self.draw_label
            self.draw_image()

        if event.key == "m":
            self.draw_prediction = not self.draw_prediction
            self.draw_image()

        if event.key == "c":
            last_poly = self.polygons[self.im_idx - 1]
            if len(last_poly[-1]) > 0:
                self.cur_polygons = deepcopy(last_poly)
            self.draw_image()

        if event.key == "1":
            self.band = self.band+1 if self.band<len(self.layers)-1 else len(self.layers)-1
            self.change_image()
            self.update()

        if event.key == "2":
            self.band -=1
            self.change_image()
            self.update()

    def onClick(self, event):

        if event.button == 1:  ## Left click
            if event.inaxes:
                if len(self.cur_polygons) == 0: #if no polygons were drawn, we reset the index and points to their init value
                    #cur_polygons holds a dict with {"idx": the index, "pts": the drawn points}
                    self.cur_polygons = [{"idx": self.cur_idx, "pts": []}]

                label = self.cur_polygons[-1]
                L_idx = label["idx"]
                polygons = label["pts"]
                polygons.append([event.xdata, event.ydata])
                self.cur_polygons[-1] = {"idx": L_idx, "pts": polygons}
                self.draw_polygons()
                print(event.xdata)
                print(event.ydata)

        if event.button == 3:
            self.submit_polygon()

    def onScroll(self, event):
        self.change_image(int(event.step))

    ###########################################
    ## Image stuff
    ###########################################
    def change_image(self, step=0):

        self.polygons[self.im_idx] = self.cur_polygons
        self.im_idx = (self.im_idx + step) % len(self.filenames)
        self.cur_polygons = self.polygons[self.im_idx]
        self.draw_image()

    def submit_polygon(self):
        new_poly = {"idx": self.cur_idx, "pts": []}
        if len(self.cur_polygons) == 0:
            self.cur_polygons.append(deepcopy(new_poly))
        elif len(self.cur_polygons[-1]["pts"]) == 0:
            self.cur_polygons[-1] = deepcopy(new_poly)
        else:
            self.cur_polygons.append(deepcopy(new_poly))

    def draw_polygons(self):
        [d.remove() for d in self.drawn if self.drawn and d]
        self.drawn = []
        clrs_lines = self.clrs_lines
        clrs_fill = self.clrs_fill
        fill_alpha = self.fill
        marker_types = ["x","o","d","s"]
        #You can add more Hexadecimal color codes if necessary. There are clrs_lines * clrs_fill amount of colors
        for poly in self.cur_polygons:
            L_idx = poly["idx"]
            for i in range(len(clrs_fill)):
                if 10*(1+i) > L_idx >= 10*(i): 
                        F_idx = i
            if len(poly["pts"]) > 0:
                if L_idx >= len(clrs_lines):
                    L_idx = L_idx - len(clrs_lines)*(F_idx+1)
                polygons = np.array(poly["pts"])
                self.drawn.extend(self.ax.fill(polygons[:, 0], polygons[:, 1], clrs_fill[F_idx], alpha=fill_alpha))
                self.drawn.extend(self.ax.plot(polygons[:, 0], polygons[:, 1], color=clrs_lines[L_idx], marker=marker_types[F_idx]))
        self.ax.figure.canvas.draw_idle()

    ###########################################
    ## Drawing
    ###########################################

    def update(self):
        self.ax.figure.canvas.draw_idle()

    def save_label_images(self, plot=1):

        filenames = self.filenames
        polygons = self.polygons

        for i, fn in enumerate(filenames):

            poly = polygons[i]
            if i <= self.im_idx:

                print("saving labels for " + str(len(poly)) + " objects:" + fn)
                im = self.read_image(fn)
                labeled = self.gen_index_image(im, poly)
                if len(poly) == 0:
                    x,y,d = np.shape(im)
                    labeled = np.zeros([x,y], dtype = np.uint8)

                #if plot:
                #    plt.imshow(labeled)
                #    plt.title('labeled')
                #    plt.show()
                self.save_labeled(fn, labeled)
                print('labels saved')

    def save_current_label_images(self, plot=1):  # todo needs testing

        fn = self.filenames[self.cur_idx]

        if len(self.cur_polygons) == 0:
            pass

        print("saving labels for " + str(len(self.cur_polygons)) + " objects:" + fn)
        im = self.read_image(fn)
        labeled = self.gen_index_image(im, self.cur_polygons)
        if plot:
            plt.imshow(labeled)
            plt.title('labeled')
            plt.show()
        self.save_labeled(fn, labeled)
        print('labels saved')

    def gen_index_image(self, im, poly):

        shape = np.array(im).shape[0:2]
        label_im = np.zeros(shape, np.uint8)

        for labels in poly:
            lab_idx, poly = labels["idx"], labels["pts"]
            if len(poly) == 1:
                poly_coor = np.round(poly)[:, ::-1]
                mask = disk(poly_coor[0, 0], poly_coor[0, 1], 10)
                label_im[mask] = lab_idx + 1

            if len(poly) > 2:
                poly_coor = np.round(poly)[:, ::-1]
                mask = polygon2mask(shape, poly_coor)
                label_im[mask] = lab_idx + 1

        return label_im

    def save_labeled(self, fn, label_im):

        # Add switch if fn is tiff / H5
        save_labeled(fn, label_im,self.m.labels)


###########################################
#     Helper Functions 
###########################################
import h5py

def save_labeled(fn, label_im, label_legend=None):
    if ".jpg" in fn.lower() or ".tif" in fn.lower():
        fn_out = fn.replace('_input', '_label')
        io.imsave(fn_out, label_im)
    if ".h5" in fn.lower():
        add_dataset_hypercube(fn, label_im, dataset="labels")
        if label_legend is not None:
            add_dataset_hypercube(fn, label_legend, dataset="label_legend")
        if fn.find("_labeled_hypercube") == -1:
            os.rename(fn, fn.replace("_hypercube", "_labeled_hypercube"))

def read_H5(fn, dataset="mask_data"):
    with h5py.File(fn, 'r') as fh:
        if dataset not in fh.keys():
            return None
        data = np.array(fh[dataset][:])

    if data.dtype == np.uint8:
        pass
    else:
        if data.max() <= 1:  data = (data * 255).astype(np.uint8)

    return data

def read_H5_hypercube(fn, dataset="data", all = []):
    with h5py.File(fn, 'r') as fh:
        if dataset not in fh['hypercube'].keys():  # ammended to read the hypercube format
            return None
        if all == []:
            data = np.array(fh['hypercube'][dataset][:])
        else:
            data = np.array(fh['hypercube'][dataset][:,:,all])

    if data.dtype == np.uint8:
        pass
    else:
        if data.max() <= 1:  data = (data * 255).astype(np.uint8)

    return data

def read_image(self, filename):
    im = cv2.imread(filename)
    return im

def channels2rgb(pd):
    clrmap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])

    if pd.shape[2] == 1:
        pd = np.tile(pd, (1, 1, 3))
    elif pd.shape[2] == 3:
        pd = pd
    else:
        pd = np.dot(pd, clrmap[:pd.shape[2]])

    pd = (pd * 255).astype(np.uint8)

    return pd

def add_dataset(fn, data, dataset="labels"):
    with h5py.File(fn, 'r+') as fh:
        if dataset in fh.keys():
            del fh[dataset]
            fh[dataset] = data
        else:
            fh.create_dataset(dataset, data=data, compression="lzf")
    fn.close()

def add_dataset_hypercube(fn, data, dataset="labels"):
    with h5py.File(fn, 'r+') as fh:
        if dataset in fh['hypercube'].keys():
            del fh['hypercube'][dataset]
            fh['hypercube'][dataset] = data
        else:
            if dataset=="label_legend":
                fh['hypercube'].create_dataset(dataset, data=data, dtype="S10")
            else:
                fh['hypercube'].create_dataset(dataset, data=data, compression="lzf")
    fh.close()

def resize_label(fn):
    lbl = read_H5(fn, dataset="labels")
    if lbl is None:
        return

    msk = read_H5(fn, dataset="mask_data")
    lbl = cv2.resize(lbl, msk.shape[::-1])
    add_dataset(fn, lbl, dataset="labels")

def resize_label_hypercube(fn):
    lbl = read_H5_hypercube(fn, dataset="labels")
    if lbl is None:
        return

    msk = read_H5_hypercube(fn, dataset="data")
    lbl = cv2.resize(lbl, msk.shape[::-1])
    add_dataset_hypercube(fn, lbl, dataset="labels")

###########################################
##          Video Frame Selecting 
###########################################

def select_video_frames(filename, out_dir="./training"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cap = cv2.VideoCapture(filename)

    fn = filename.split("\\")[-1].split(".")[0]

    f_num = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("", frame)
        quit = check_keys(out_dir, fn, f_num, frame)
        if quit: break

        f_num += 1

    cap.release()
    cv2.destroyAllWindows()  # destroy all the opened windows


def select_dataset_frames(filename, data, out_dir="./training"):
    if not os.path.exists(out_dir):   os.makedirs(out_dir)

    fn = filename.split("\\")[-1].split(".")[0]

    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("", 900, 600)

    rate = 1000

    for t in range(len(data)):

        if t < 5000: continue
        if t > 50000: continue
        if np.mod(t, rate) > 0: continue

        im = data[t]
        im = norm_uint8(im)

        cv2.imshow("", im)
        save_frame(out_dir, fn, im, t)

        quit = check_keys(out_dir, fn, im, t)
        if quit: break

    cv2.destroyAllWindows()  # destroy all the opened windows


def save_frames(out_dir, fn, data, t):
    im = data[t]
    im = norm_uint8(im)
    fn_out = f"{out_dir}/{fn}_f{t}_input.tiff"
    cv2.imwrite(fn_out, im)


def norm_uint8(im):
    im -= np.min(im)
    im /= (np.max(im) * 255)
    im = im.astype(np.uint8)

    return im


def save_input_frame(out_dir, fn, t):
    read_frame(fn, t)


def read_frame(fn, n):
    reader = cv2.VideoCapture(fn)
    len_frames = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    reader.set(1, n)
    ret, im = reader.read()
    return im


def save_frame(out_dir, fn, im, t):
    fn_out = f"{out_dir}/{fn}_f{t}_input.tiff"
    cv2.imwrite(fn_out, im)
    print(fn_out)


def check_keys(out_dir, fn, im, t):
    key = cv2.waitKeyEx(1)
    quit = False
    if key > 0:
        if key == ord('q'):
            quit = True
        elif key == ord('s'):
            save_frame(out_dir, fn, im, t)

    return quit


############################################
############################################      


if __name__ == "__main__":
    import os
    import glob
    an = Annotate(exclude_labeled=True, run=True, bands=[1])
    plt.show()
    an.save_label_images()