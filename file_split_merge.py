from filesplit.merge import Merge
from filesplit.split import Split
import os, glob

path = os.getcwd().replace("\\", "/")

def fsplit_fun():
    for i in glob.glob("*.pkl"):
        os.mkdir(i.strip(".pkl"))
        split = Split(i, i.strip(".pkl"))
        split.bysize(size = 1024*1000*20)
        
def fmerge_fun(files):
    for i in files:
        if not os.path.exists(path+"/"+i+".pkl"):
            merge = Merge(path+"/"+i, path, i+".pkl")
            merge.merge()

#fsplit_fun()
#fmerge_fun(["model1"])
