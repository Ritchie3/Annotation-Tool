
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image 
from skimage.draw import polygon2mask, circle
import os 


class Annotate():
    def __init__(self,filenames):
        
        fig, ax = plt.subplots(figsize=(10,10))
    
        self.fig = fig
        self.ax = ax
        self.im_idx = 0
        self.cur_idx = 0
        self.filenames = filenames
        self.drawn = []
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.polygons = [[],[]] * len(filenames)
        self.cur_polygons = [[[],[]]]
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
            if len(self.cur_polygons[-1][1])>0:
                self.cur_polygons.append([[],[]])
            self.draw_image()
        if event.key == 'down':
            self.cur_idx = max(self.cur_idx-1,0)
            if len(self.cur_polygons[-1][1])>0:
                self.cur_polygons.append([[],[]])
            self.draw_image()
        if event.key == 'enter':
            if len(self.cur_polygons[-1][1])>0:
                self.cur_polygons.append([[],[]])
        if event.key == 'escape':
            self.polygons[self.im_idx] = self.cur_polygons
            plt.close(self.fig)
            
        if event.key == 'backspace':

            if len(self.cur_polygons[-1][1])>0 :
                self.cur_polygons[-1][1].pop()
            elif len(self.cur_polygons)>1:
                self.cur_polygons.pop()
            self.draw_polygons()

    def onClick(self,event):
        if event.inaxes:
            L_idx, polygons = self.cur_polygons[-1]
            if not L_idx: 
                L_idx = self.cur_idx

            polygons.append( [event.xdata ,event.ydata] )
            self.cur_polygons[-1] = [L_idx, polygons]
            self.draw_polygons()
            
    def draw_polygons(self):
        [d.remove() for d in self.drawn if self.drawn and d]
        self.drawn = []
        clrs = "rgbcym"
        for poly in self.cur_polygons:
            if poly[1]:
                L_idx = poly[0]
                polygons = np.array(poly[1])
                self.drawn.extend( self.ax.fill(polygons[:,0],polygons[:,1], clrs[L_idx], alpha=0.4) )
                self.drawn.extend( self.ax.plot(polygons[:,0],polygons[:,1], "-o"+clrs[L_idx] ))
        self.ax.figure.canvas.draw_idle()

    def save_label_images(self):

        filenames = self.filenames
        polygons = self.polygons

        for i, fn in enumerate(filenames):
            IM = Image.open(fn)
            shape = np.array(IM).shape[0:2]
            label_im = np.zeros(shape)
            for lab_idx, poly in polygons[i]:
                
                if len(poly)==1:
                    poly_coor = np.round(poly)[:,::-1]
                    mask = circle(poly_coor[0,0], poly_coor[0,1], 10)
                    label_im[mask] = lab_idx+1

                if len(poly)>2:
                    poly_coor = np.round(poly)[:,::-1]
                    mask = polygon2mask(shape, poly_coor)
                    label_im[mask] = lab_idx+1       
            
            fn = "../ML Training/Flap_input.jpg"
            fn_out = fn.replace('_input','_label')    
            plt.imsave(fn_out , label_im)

            #plt.figure()
            #plt.imshow(label_im)
            #plt.figure()
            #plt.imshow(IM) 
        
        
