import numpy as np
import pandas as pd
import cam_communicator_class as cc
import time

### change settings here  #
nx, ny = (680, 680)
slide = 1
jobname = "Pattern16"
ipaddress = '127.0.0.1'
filename = 'selected_cells.csv'
#ipaddress = '10.11.113.30'
# sign is opposite of what it is in CellProfiler
offset_X = 0 #-45
offset_Y = 0 #+49
time_between_commands = 0.5

################
# Open communication channel to Leica

sp5=cc.CAMcommunicator()
sp5.setIP(ipaddress)
sp5.open()



sp5.flushCAMreceivebuffer()
time.sleep(time_between_commands)
sp5.deleteCAMList()
time.sleep(time_between_commands)

objects = pd.read_csv(filename)

# LAS AF Matrix Screener expects pixel coordinates relative to the centre
# subtract half the image width and height
objects.Location_Center_X -= nx/2.0 + offset_X
objects.Location_Center_Y -= ny/2.0 + offset_Y

objects.Location_Center_X = objects.Location_Center_X.astype(int)
objects.Location_Center_Y = objects.Location_Center_Y.astype(int)

for i, row in objects.iterrows():
    sp5.addJobToCAMlist(jobname, row.Location_Center_X, row.Location_Center_Y, slide, row.Metadata_U+1, row.Metadata_V+1, row.Metadata_X+1, row.Metadata_Y+1)
    time.sleep(time_between_commands)


sp5.startCAMScan()
sp5.stopWaitingForCAM()

