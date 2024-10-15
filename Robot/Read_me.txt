‘rotation’ is the rotation control program of robotic arm using shear force.
‘test’ is the translation control program of robotic arm using normal force.
'SerialPortSensorApp_4FPC' is the developed MATLAB app for visualization. 

In the interaction process, sensor signals are send to MATLAB via serial port, then the trained model was called in MATLAB for localization and force estimation. Finally the control signals are send to robotic arms though the control program.