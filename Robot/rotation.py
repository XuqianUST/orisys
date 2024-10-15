import urx
import numpy as np
import serial

rob = urx.Robot("192.168.1.10")

ser = serial.Serial("COM2",baudrate=9600,timeout=1)

# rob.movej([
#     np.radians(-182.4),
#     np.radians(-69.32),
#     np.radians(-126.6),
#     np.radians(-142.5),
#     np.radians(178.52),
#     np.radians(8.75),
# ],acc=0.1,vel=0.2,relative=False)

rob.movej([
    np.radians(-181.8),
    np.radians(-69.2),
    np.radians(-124.9),
    np.radians(-166.2),
    np.radians(269.6),
    np.radians(-27.6),
],acc=0.1,vel=0.2,relative=False)

while 1:
    try:
        len = 0
        raw_data = ser.read(8)
        decoded_data = raw_data.decode('ascii')
        decoded_data = decoded_data.strip()


        data = np.round(int(decoded_data))
        print("debug1",data)

        if data > 1000:
            len = 0.05
        elif data < -1000:
            len = -0.05

        print(len)

        rob.movel([len,0,0,0,0,0],acc=0.1,vel=0.2,relative=True)

        print("success")
    
    except:
        rob.movej([0,0,0,0,0,0],acc=0.1,vel=0.05,relative=True)
        continue


rob.close()