import os
import sys
import time
from datetime import datetime
import warnings
import time

warnings.simplefilter("ignore")

if __name__=="__main__":
    start = datetime.now()
    print("\n\n CROSS DATA DRIFT COMPARISIONS STARTED AT ", start, "\n")
    data_list = ["HOPN","ICDE","HASOC19"]
    print(data_list)

    for base in data_list:
        for new in data_list:
             if base==new:
                 continue
            print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")
            print("TRAIN on {} and TEST on {}".format(base,new))
            os.system('python3 cross_dataset_drift.py --base {} --new {}'.format(base, new))
            print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")
    end = datetime.now()
    print("\n\n CROSS DRIFT ", end)
    print("\n\n Total time taken ", end-start)
    print("-- DONE --")
