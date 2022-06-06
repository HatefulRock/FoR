from distutils.log import debug
import time
import math
import matplotlib.pyplot as plt
import numpy as np


def theta_s(x,y):
    if x>0:
        return 1*math.atan(1*y)
    if x<=0:
        return 1*math.atan(-1*y)

class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN
        self.J=[[0,0],[0,0]] #Jacobienne
        self.JG=[[0,0],[0,0]] #Jacobienne avec Centre gravité
        self.coordG=[0,0]

        self.alpha = [1/4,1/4,1/(math.pi)]  # normalition avec limite du monde cartesien = -1m à + 1m

    def train(self, target, show_animation=True):
        position = self.robot.get_position()

        network_input = [0, 0]
        network_input[0] = (target[0]-position[0])*self.alpha[0]
        network_input[1] = (target[1]-position[1])*self.alpha[1]
        #Teta_t = 0

        while self.running:
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t
            delta_ti=0.010
            Keps=0.95
            KG=0.05          
            alpha_x = 1/4
            alpha_y = 1/4
            self.JG[0][0]=(self.robot.M1*self.robot.L1*math.sin(position[2])+self.robot.M2*self.robot.L2*math.sin(position[2]+position[3]))/(-2*(self.robot.M1+self.robot.M2+self.robot.M3))
            self.JG[1][0]=(self.robot.M1*self.robot.L1*math.cos(position[2])+self.robot.M2*self.robot.L2*math.cos(position[2]+position[3]))/(2*(self.robot.M1+self.robot.M2+self.robot.M3))
            self.JG[0][1]=-self.robot.M2*self.robot.L2*math.sin(position[2]+position[3])/(2*(self.robot.M1+self.robot.M2+self.robot.M3))
            self.JG[0][1]=self.robot.M2*self.robot.L2*math.cos(position[2]+position[3])/(2*(self.robot.M1+self.robot.M2+self.robot.M3))
            self.J[0][0]=-self.robot.L1*math.sin(position[2])-self.robot.L2*math.sin(position[2]+position[3])
            self.J[0][1]=-self.robot.L2*math.sin(position[2]+position[3])
            self.J[1][0]=self.robot.L1*math.cos(position[2])+self.robot.L2*math.cos(position[2]+position[3])
            self.J[0][1]=self.robot.L2*math.cos(position[2]+position[3])
                        
            crit_av= alpha_x*alpha_x*(position[0]-target[0])*(position[0]-target[0]) + alpha_y*alpha_y*(position[1]-target[1])*(position[1]-target[1]) 
            
            self.robot.set_motor_velocity(command) # applique vitesses roues instant t,                     
            position = self.robot.get_position() #  obtient nvlle pos robot instant t+1 
            self.coordG[0]=(self.robot.M1*self.robot.L1*math.cos(position[2])+self.robot.M2*self.robot.L2*math.cos(position[2]+position[3])+self.robot.M3*target[0])/(2*(self.robot.M1+self.robot.M2+self.robot.M3))
            self.coordG[1]=(self.robot.M1*self.robot.L1*math.sin(position[2])+self.robot.M2*self.robot.L2*math.sin(position[2]+position[3])+self.robot.M3*target[1])/(2*(self.robot.M1+self.robot.M2+self.robot.M3))      
            
            wrist=plot_arm(position[2],position[3],target[0],target[1],self.robot.L1,self.robot.L2,delta_ti, show_animation)
            network_input[0] = (target[0]-position[0])*self.alpha[0]
            network_input[1] = (target[1]-position[1])*self.alpha[1]
            
            crit_ap= alpha_x*alpha_x*(position[0]-target[0])*(position[0]-target[0]) + alpha_y*alpha_y*(position[1]-target[1])*(position[1]-target[1]) 

            if self.training:
                delta_t = 0.050

                grad = [
                    -2*delta_t*Keps*(alpha_x*alpha_x*(position[0]-target[0])*self.J[0][0]+alpha_y*alpha_y*(position[1]-target[1])*self.J[1][0])+2*delta_t*KG*(self.coordG[0]*self.JG[0][0]+self.coordG[1]*self.JG[1][0]),

                    -2*delta_t*Keps*(alpha_x*alpha_x*(position[0]-target[0])*self.J[0][1]+alpha_y*alpha_y*(position[1]-target[1])*self.J[1][1])+2*delta_t*KG*(self.coordG[0]*self.JG[0][1]+self.coordG[1]*self.JG[1][1])
                    ]

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap <= crit_av) :
                    self.network.backPropagate(grad, 0.2,0) # grad, pas d'app, moment
                else :
                    #self.network.random_update(0.001)
                    self.network.backPropagate(grad, 0.2, 0)
                
        self.robot.set_motor_velocity([0,0]) # stop  apres arret  du prog d'app
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
                #Teta_t=position[2]
             
                
        
        self.running = False

def plot_arm(theta1, theta2, target_x, target_y, l1, l2, dt, show_animation):  # pragma: no cover
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + \
        np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    if show_animation:
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')

        plt.plot([wrist[0], target_x], [wrist[1], target_y], 'g--')
        plt.plot(target_x, target_y, 'g*')

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.show()
        plt.pause(dt)

    return wrist
