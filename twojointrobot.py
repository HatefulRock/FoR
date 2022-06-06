import time
import math
import matplotlib.pyplot as plt
import numpy as np

class Rob:
    def __init__(self,L1,L2,q1,q2):
        self.L1=L1
        self.L2=L2
        self.q1=q1
        self.q2=q2
        self.Xp=self.get_cartecoord()[0]
        self.Yp=self.get_cartecoord()[1]
        self.q1point=0
        self.q2point=0
        self.deltat=0.050
    
    def get_position(self):
        position=[self.Xp,self.Yp,self.q1,self.q2]
        return(position)
    
    def set_motor_velocity(self,command):
        self.q1point=command[0]*math.pi/6
        self.q2point=command[1]*math.pi/6
        time.sleep(self.deltat)
        self.q1+=self.q1point*self.deltat
        self.q2+=self.q2point*self.deltat
        self.Xp=self.get_cartecoord()[0]
        self.Yp=self.get_cartecoord()[1]
        print(self.Xp,self.Yp)
    
    def get_cartecoord(self):
        Xp=self.L1*math.cos(self.q1)+self.L2*math.cos(self.q1+self.q2)
        Yp=self.L1*math.sin(self.q1)+self.L2*math.sin(self.q1+self.q2)
        cartecoord=[Xp,Yp]
        return(cartecoord)
    


    

