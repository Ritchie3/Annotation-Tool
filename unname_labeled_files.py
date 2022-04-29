import GUI_functions
import glob
import os

folder = GUI_functions.GUIgetdir()
searchpath = folder + r'/*h5'
print(searchpath)
filenames = glob.glob(searchpath)
for fn in filenames:
    if "_labeled_hypercube" in fn:
        print(fn)
        os.rename(fn, fn.replace("_labeled_hypercube","_hypercube"))

