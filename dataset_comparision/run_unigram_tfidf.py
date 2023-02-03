import os
import sys
import time
from datetime import datetime
import warnings
import time

warnings.simplefilter("ignore")

if __name__=="__main__":
    INPUT_PATH = ""
    start = datetime.now()
    print("\n\n DATA DRIFT COMPARISIONS STARTED AT ", start, "\n")
    data_list = os.listdir(INPUT_PATH)

    if ".ipynb_checkpoints" in data_list:
        data_list.remove(".ipynb_checkpoints")

    print(data_list)

   
    for di in data_list:
        print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")
        print("Dataset {}".format(di))
        os.system('python3 unigram_tfidf.py --base {} --ng {}'.format(di, 3))
        print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")
    end = datetime.now()
    print("\n\n UNIGRAM TFIDF ", end)
    print("\n\n Total time taken ", end-start)
    print("-- DONE --")
