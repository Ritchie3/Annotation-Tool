
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image 
from skimage.draw import polygon2mask, circle
from skimage import io
import os 

from copy import deepcopy


class Annotate():
    def __init__(self,filenames, exclude_labeled=True):

        plt.rcParams['keymap.fullscreen'] = []
        
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

        self.polygons =  [[] for i in range(len(filenames))]
        self.cur_polygons = []

        self.open_window()


    def open_window(self):
        fig, ax = plt.subplots(figsize=(18,15))
        self.fig = fig
        self.ax = ax
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        
        self.draw_image()

    def draw_image( self ):
        filename = self.filenames[self.im_idx]
        self.ax.clear()

        im = io.imread(filename) 
        self.ax.imshow(im, cmap=self.cmap)  

        instructions =  "Left: Last Image                    Right: Next Image \n" + \
                        "Up: Iterate Index up                Down: Iterate Index Down \n" + \
                        "Enter/Right Click: Next Object      Backspace: Remove Point\n" + \
                        "Escape: Close Tool \n"

        self.ax.set_title( instructions , fontsize = 20  )
        self.ax.set_xlabel(filename + " : Label " + str(self.cur_idx),  fontsize = 20 )
        self.draw_polygons()
        
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

    def save_label_images(self):

        filenames = self.filenames
        polygons = self.polygons
        print("x")
        for i, fn in enumerate(filenames):


            if len(polygons[i])==0:
                print(fn)
                continue

            im = io.imread(fn) 
            shape = np.array(im).shape[0:2]
            label_im = np.zeros(shape, np.uint8)           

            for labels in polygons[i]:
                lab_idx, poly = labels["idx"],labels["pts"]
                if len(poly)==1:
                    poly_coor = np.round(poly)[:,::-1]
                    mask = circle(poly_coor[0,0], poly_coor[0,1], 10)
                    label_im[mask] = lab_idx+1

                if len(poly)>2:
                    poly_coor = np.round(poly)[:,::-1]
                    mask = polygon2mask(shape, poly_coor)
                    label_im[mask] = lab_idx+1      
            print(fn) 
            
            fn_out = fn.replace('_input','_label')    
            io.imsave(fn_out , label_im)

        
        
