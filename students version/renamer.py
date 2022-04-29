import os
import glob
place = r"C:\Users\gille\OneDrive\Documenten\gilles mapppeke\Machine Learning\data\hypercubes\hypercube_cut"
filenames = glob.glob(place + r'/*h5')

to_displace = "labeled"

for i, fn in enumerate(filenames):
    if fn.find(to_displace)!=-1:
        fn = fn.replace("_"+ to_displace + "_hypercube", "_hypercube")
        print(fn)
        print(filenames[i])
        os.rename(filenames[i], fn)
