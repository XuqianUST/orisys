import urx
import numpy as np
import serial

rob = urx.Robot("192.168.1.10")

ser = serial.Serial("COM2",baudrate=9600,timeout=1)

rob.movej([
    np.radians(-182.4),
    np.radians(-80.2),
    np.radians(-103.4),
    np.radians(-151.79),
    np.radians(178.52),
    np.radians(14.65),
],acc=0.1,vel=0.2,relative=False)

while 1:
    try:
        raw_data = ser.read(8)
        decoded_data = raw_data.decode('ascii')
        decoded_data = decoded_data.strip()

        print(decoded_data)

        theta = np.radians(np.round(int(decoded_data)))

        print(theta)

            # if float(fine_data)>105:
            #     theta = np.radians(np.round(float(fine_data))-100)
            #     print("debug1")
            # else:
            #     theta = np.radians(np.round(float(fine_data)))
            #     print("debug2")

        r = 0.015

        z,x = -r*np.cos(theta-np.pi/6), r*np.sin(theta-np.pi/6)

        rob.movel([3*x,0,z,0,0,0],acc=0.1,vel=0.05,relative=True)
        print("success")
    
    except:
        rob.movel([0,0,0,0,0,0],acc=0.1,vel=0.05,relative=True)
        continue


rob.close()