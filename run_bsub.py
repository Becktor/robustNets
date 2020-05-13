import time
import os
import subprocess

for i in range(4):
    x = 'bsub < robust_net.sh'
    os.system(x)
    pending = True
    while(pending==True):
        outString = subprocess.check_output("bstat")
        if b'PEND' in outString:
            time.sleep(60)
        else:
            pending = False
	time.sleep(86500)
