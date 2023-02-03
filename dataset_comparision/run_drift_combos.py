import os
import sys
import time
from datetime import datetime
import warnings
import time

warnings.simplefilter("ignore")

if __name__=="__main__":
    INPUT_PATH = "" "" ## ADD YOUR PATH HERE
    start = datetime.now()
    print("\n\n DATA DRIFT COMPARISIONS STARTED AT ", start, "\n")
    data_list = os.listdir(INPUT_PATH)

    if ".ipynb_checkpoints" in data_list:
        data_list.remove(".ipynb_checkpoints")

    print(data_list)

    l = len(data_list)
    print("\n TOTAL COMBOS:: ", str((l*(l+1)/2)))
    combo_number = 1
    for i in range(l):
        d1=data_list[i]
        for j in range(i+1,l):
            d2=data_list[j]
            print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")
            print("COMBO # {} PAIR {} ::--:: {}".format(combo_number, d1, d2))
            combo_number+=1
            os.system('python3 data_drift.py --base {} --new {}'.format(d1, d2))
            print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")
    end = datetime.now()
    print("\n\n DATA DRIFT COMPARISIONS ENDED AT ", end)
    print("\n\n Total time taken ", end-start)
    print("-- DONE --")
