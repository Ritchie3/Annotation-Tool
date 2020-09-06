
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image 
from skimage.draw import polygon2mask, circle
from skimage import io
import os 


class Annotate():
    def __init__(self,filenames):
        
        filenames = [fn for fn in filenames if not os.path.exists(fn.replace("_input","_label"))]
        self.filenames = filenames

        fig, ax = plt.subplots(figsize=(18,15))
        self.fig = fig
        self.ax = ax

        self.im_idx = 0
        self.cur_idx = 0
        
        self.drawn = []
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.polygons = [[]] * len(filenames)
        self.cur_polygons = []
        self.draw_image()
        
    def draw_image( self ):
        filename = self.filenames[self.im_idx]
        self.ax.clear()
        self.ax.imshow(Image.open(filename))   
        self.ax.set_title("Left: Last Image        Right: Next Image \n" + 
                          "Up: Iterate Index up    Down: Iterate Index Down \n" + 
                          "Enter: Next Object      Backspace: Remove Point\n" + 
                          "Escape: Close Tool \n"   )
        self.ax.set_xlabel(filename + " : Label " + str(self.cur_idx))
        self.draw_polygons()
        
    def onKey(self, event):
        if event.key == 'right':

            self.polygons[self.im_idx] = self.cur_polygons
            self.im_idx = (self.im_idx +1) % len(self.filenames)
            self.cur_polygons = self.polygons[self.im_idx] 
            self.draw_image()  
        if event.key == 'left':
            self.polygons[self.im_idx] = self.cur_polygons
            self.im_idx = (self.im_idx -1) % len(self.filenames)
            self.cur_polygons = self.polygons[self.im_idx]
            self.draw_image()

        if event.key == 'up':
            self.cur_idx += 1
            new_poly = {"idx":self.cur_idx, "pts":[]}
            if len(self.cur_polygons[-1]["pts"])>0:
                self.cur_polygons.append(new_poly)
            else:
                self.cur_polygons[-1] = new_poly
            self.draw_image()    

        if event.key == 'down':
            self.cur_idx = max(self.cur_idx-1,0)
            new_poly = {"idx":self.cur_idx, "pts":[]}
            if len(self.cur_polygons[-1]["pts"])>0:
                self.cur_polygons.append(new_poly)
            else:
                self.cur_polygons[-1] = new_poly
            self.draw_image()

        if event.key == 'enter':
            new_poly = {"idx":self.cur_idx, "pts":[]}
            if len(self.cur_polygons[-1]["pts"])>0:
                self.cur_polygons.append(new_poly)
            else:
                self.cur_polygons[-1] = new_poly

        if event.key == 'escape':
            self.polygons[self.im_idx] = self.cur_polygons
            plt.close(self.fig)
            
        if event.key == 'backspace':
            if len(self.cur_polygons[-1]["pts"])>0 :
                self.cur_polygons[-1]["pts"].pop()
            elif len(self.cur_polygons)>1:
                self.cur_polygons.pop()
            self.draw_polygons()

    def onClick(self,event):
        if event.inaxes:

            if len(self.cur_polygons)==0:
                self.cur_idx = 0
                self.cur_polygons = [{"idx":self.cur_idx,"pts":[] }]

            label = self.cur_polygons[-1]
            L_idx = label["idx"]
            polygons = label["pts"]
            polygons.append( [event.xdata ,event.ydata] )
            self.cur_polygons[-1] = {"idx":L_idx, "pts":polygons}

            self.draw_polygons()
            
    def draw_polygons(self):
        [d.remove() for d in self.drawn if self.drawn and d]
        self.drawn = []
        clrs = "rgbcym"
        for poly in self.cur_polygons:
            if len(poly["pts"])>0:
                L_idx = poly["idx"]
                polygons = np.array(poly["pts"])
                self.drawn.extend( self.ax.fill(polygons[:,0],polygons[:,1], clrs[L_idx], alpha=0.2) )
                self.drawn.extend( self.ax.plot(polygons[:,0],polygons[:,1], "-o"+clrs[L_idx] ))
        self.ax.figure.canvas.draw_idle()

    def save_label_images(self):

        filenames = self.filenames
        polygons = self.polygons

        for i, fn in enumerate(filenames):
            IM = Image.open(fn)
            shape = np.array(IM).shape[0:2]
            label_im = np.zeros(shape, np.uint8)
            if len(polygons[i])==0:
                print(fn)
                continue

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
            
            fn_out = fn.replace('_input','_label')    
            io.imsave(fn_out , label_im)

            #plt.figure()
            #plt.imshow(label_im)
            #plt.figure()
            #plt.imshow(IM) 
        
        
