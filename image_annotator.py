
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image 
from skimage.draw import polygon2mask, circle
from skimage import io
import os 

from copy import deepcopy

import cv2 

###############################
#  Annotator 
###############################

class Annotate():

    def __init__(self,filenames, exclude_labeled=True, run=True):

        if exclude_labeled:
            filenames = [fn for fn in filenames if not os.path.exists(fn.replace("_input","_label"))]
        
        if len(filenames)==0:
            print("no images")
            return

        self.filenames = filenames
        self.im_idx = 0
        self.cur_idx = 0
        self.cmap = "gray"
        self.fill = 0.25
        self.drawn = []

        self.model = None
        self.draw_prediction = False

        self.polygons =  [[] for i in range(len(filenames))]
        self.cur_polygons = []

        if run:
            self.open_window()

    def open_window(self):

        plt.rcParams['keymap.fullscreen'] = []
        fig, ax = plt.subplots(figsize=(18,15))
        self.fig = fig
        self.ax = ax
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        self.draw_image()

    def read_image(self,filename):

        ## Add Switch
        im = cv2.imread(filename) 

        return im
    
    def load_data(self):

        filename = self.filenames[self.im_idx]

        im = self.read_image(filename)

        if im is None:
            print( str(self.im_idx) + ": Not Found" )
            self.filenames.pop(self.im_idx)
            self.polygons.pop(self.im_idx)

            im = self.load_data()
        
        if im.dtype==np.bool:        im = im*255
        if im.ndim == 2:             im = im[...,None]
        

        im = im.astype(np.uint8)

        return im 

    def draw_image( self ):
        
        im = self.load_data()

        self.ax.clear()
        
        if self.model is not None:
            if self.draw_prediction:
                
                im_in = np.mean(im,axis=2) 
                pred = self.model.segment(im_in)

                im = np.mean(im,axis=2)           
                im = im[...,None]*[1,1,1]       

                pred = pred.transpose([1,2,0])

                im = channels2rgb(pred)

                #
        if im.shape[-1]==1:          im = im*[1,1,1]
        im = im.astype(np.uint8)
        self.ax.imshow(im)      

        instr = [["Left: Last Image","Right: Next Image" ],
                        ["Up: Iterate Index up","Down: Iterate Index Down"],
                        ["Backspace: Remove Point","Enter/Right Click: Next Object"],
                        ["Escape: Close Tool","[i] : More information"]]

        # Add a table at the bottom of the axes
        table = plt.table(cellText= instr, loc='top')
        table.set_fontsize(30)
        table.scale(1,3)

        fn_str = self.filenames[self.im_idx]
        fn_str = ("..."+fn_str[-20:] if len(fn_str) > 20 else fn_str)

        #self.ax.set_title( instructions , fontsize = 20  )
        self.ax.set_xlabel(fn_str + "\n Label " + str(self.cur_idx),  fontsize = 40 )
        self.draw_polygons()
    ##  UI   
    def onKey(self, event):
        if event.key == 'right':
            self.change_image(1)

        if event.key == 'left':
            self.change_image(-1)

        if event.key == 'up':
            self.cur_idx += 1
            self.submit_polygon()
            self.draw_image()    

        if event.key == 'down':
            self.cur_idx = max(0, self.cur_idx-1)
            self.submit_polygon()
            self.draw_image()

        if event.key == 'enter':
            self.submit_polygon()

        if event.key == 'escape':
            self.polygons[self.im_idx] = self.cur_polygons
            plt.close(self.fig)
            
        if event.key == 'backspace':
            if len(self.cur_polygons[-1]["pts"])>0 :
                self.cur_polygons[-1]["pts"].pop()
            elif len(self.cur_polygons)>1:
                self.cur_polygons.pop()
            self.draw_polygons()

        if event.key == "f":
            self.fill = (self.fill + 0.25)%1
            self.draw_image( )

        if event.key == "j":
            self.cmap = "jet"
            self.draw_image( )

        if event.key == "g":    
            self.cmap = "gray"
            self.draw_image( )

        if event.key == "m":
            self.draw_prediction = not self.draw_prediction
            self.draw_image( )

        if event.key == "c":
            last_poly = self.polygons[self.im_idx-1]
            if len(last_poly[-1])>0:
                self.cur_polygons = deepcopy(last_poly)
            self.draw_image( )

    def onClick(self,event):

        if event.button==1: ## Left Button 
            if event.inaxes:

                if len(self.cur_polygons)==0:
                    #self.cur_idx = 0
                    self.cur_polygons = [{"idx":self.cur_idx,"pts":[] }]

                label = self.cur_polygons[-1]
                L_idx = label["idx"]
                polygons = label["pts"]
                polygons.append( [event.xdata ,event.ydata] )
                self.cur_polygons[-1] = {"idx":L_idx, "pts":polygons}

                self.draw_polygons()

        if event.button==3:
            self.submit_polygon()

    def onScroll(self,event):
        self.change_image(int(event.step))
    ## Image stuff 
    def change_image(self,step):

        self.polygons[self.im_idx] = self.cur_polygons
        self.im_idx = (self.im_idx+step) % len(self.filenames)
        self.cur_polygons = self.polygons[self.im_idx] 
        self.draw_image()  

    def submit_polygon(self):
        new_poly = {"idx":self.cur_idx, "pts":[]}

        if len(self.cur_polygons)==0:
            self.cur_polygons.append(deepcopy(new_poly))
        elif len(self.cur_polygons[-1]["pts"])==0:
            self.cur_polygons[-1] = deepcopy(new_poly)
        else:
            self.cur_polygons.append(deepcopy(new_poly))
            
    def draw_polygons(self):
        [d.remove() for d in self.drawn if self.drawn and d]
        self.drawn = []

        fill_alpha = self.fill
        clrs = "rgbcym"
        for poly in self.cur_polygons:
            if len(poly["pts"])>0:
                L_idx = poly["idx"]
                polygons = np.array(poly["pts"])
                
                self.drawn.extend( self.ax.fill(polygons[:,0],polygons[:,1], clrs[L_idx], alpha=fill_alpha) )
                self.drawn.extend( self.ax.plot(polygons[:,0],polygons[:,1], "-o"+clrs[L_idx] ))
        self.ax.figure.canvas.draw_idle()
    
    ## Drawing 
    def save_label_images(self):

        filenames = self.filenames
        polygons = self.polygons
        
        for i, fn in enumerate(filenames):

            poly = polygons[i]

            if len(poly)==0:  
                continue

            print("saving labels for "+ str(len(poly)) + " objects:" + fn) 
            im = self.read_image(fn)
            labeled = self.gen_index_image(im,poly)
            self.save_labeled( fn, labeled )

    def gen_index_image(self,im,poly):

        shape = np.array(im).shape[0:2]
        label_im = np.zeros(shape, np.uint8)           

        for labels in poly:
            lab_idx, poly = labels["idx"],labels["pts"]
            if len(poly)==1:
                poly_coor = np.round(poly)[:,::-1]
                mask = circle(poly_coor[0,0], poly_coor[0,1], 10)
                label_im[mask] = lab_idx+1

            if len(poly)>2:
                poly_coor = np.round(poly)[:,::-1]
                mask = polygon2mask(shape, poly_coor)
                label_im[mask] = lab_idx+1  

        return  label_im

    def save_labeled(self,fn, label_im):    

        #Add switch if fn is tiff / H5
        save_labeled(fn, label_im)
        
###########################################
#     Helper Functions 
###########################################

def save_labeled(fn, label_im):
    fn_out = fn.replace('_input','_label')    
    io.imsave(fn_out , label_im)

def read_H5(fn):
    dataset = "mask_data"
    with h5py.File(fn, 'r') as fh:   
        if dataset not in fh.keys(): 
            return None
        data = np.array(  fh[dataset][:]  )       
    return data

def add_dataset(fn, data, dataset="labels"):
    with h5py.File(fn, 'r+') as fh:
        if dataset in fh.keys():  
            del fh[dataset]
            fh[dataset] = data
        else:   
            fh.create_dataset(dataset, data=data , compression="lzf")

def read_image(self,filename):
    im = cv2.imread(filename) 
    return im

def channels2rgb(pd):

    clrmap = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]])

    if   pd.shape[2] == 1:     pd = np.tile(pd,(1,1,3))
    elif pd.shape[2] == 3:     pd = pd
    else: pd = np.dot(pd, clrmap[:pd.shape[2]])
    pd = pd*255

    return pd

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
        
        ret,frame = cap.read()
        if not ret: break
           
        cv2.imshow("",frame)
        quit = check_keys(out_dir, fn, f_num, frame)    
        if quit: break
        
        f_num +=1  
                     
    cap.release()
    cv2.destroyAllWindows()  # destroy all the opened windows
      
def select_dataset_frames(filename, data, out_dir="./training"):
     
    if not os.path.exists(out_dir):   os.makedirs(out_dir)
        
    fn = filename.split("\\")[-1].split(".")[0]
    
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("", 900, 600)   

    rate = 1000
    
    for t in range(len(data)):

        if t<5000: continue 
        if t>50000: continue 
        if np.mod(t,rate)>0: continue
        print(t)
        
        im = data[t]
        im = norm_uint8(im)
        
        cv2.imshow("",im)  
        save_frame(out_dir, fn, im, t )

        quit = check_keys(out_dir, fn, im, t)
        if quit: break
                     
    cv2.destroyAllWindows()  # destroy all the opened windows

def save_frames(out_dir,fn,data,t):

    im = data[t]
    im = norm_uint8(im)
    fn_out = f"{out_dir}/{fn}_f{t}_input.tiff"
    cv2.imwrite(fn_out, im)    
    
def norm_uint8(im):
    
    im = im*1.
    im -= np.min(im)
    im /= np.max(im)
    im = (im*255).astype(np.uint8)  
    
    return im
    
def save_input_frame(out_dir,fn,t):
    read_frame(fn, n )
    
def read_frame(fn, n ):
    reader = cv2.VideoCapture(fn)
    len_frames = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    reader.set(1, n)
    ret,im = reader.read()
    return im 

def save_frame(out_dir, fn, im, t ):
    fn_out = f"{out_dir}/{fn}_f{t}_input.tiff"
    cv2.imwrite(fn_out, im)
    print(fn_out)     


def check_keys(out_dir, fn, im, t):
    key = cv2.waitKeyEx(1)
    quit=False
    if key>0:
        if key == ord('q'):                
            quit=True
        elif key == ord('s'):
            save_frame(out_dir, fn, im, t )
                  
    return quit

############################################
# ##########################################      
        
