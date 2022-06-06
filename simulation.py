from distutils.command.sdist import show_formats
from scipy.__config__ import show
from backprop import NN
#from vrep_pioneer_simulation import VrepPioneerSimulation
#from rdn import Pioneer # rdn pour ROS avec le pioneer
#import rospy
from trainer import OnlineTrainer
from twojointrobot import Rob
import json
import threading
import matplotlib.pyplot as plt
import numpy as np

show_animation = True

if show_animation:
    plt.ion()

#robot = VrepPioneerSimulation()
robot = Rob(1,1,2.7,-0.5,1,1,1)
HL_size= 10# nbre neurons of Hiden layer
network = NN(2, HL_size, 2)

choice = input('Do you want to load previous network? (y/n) --> ')
if choice == 'y':
    with open('last_w.json') as fp:
        json_obj = json.load(fp)

    for i in range(2):
        for j in range(HL_size):
            network.wi[i][j] = json_obj["input_weights"][i][j]
    for i in range(HL_size):
        for j in range(2):
            network.wo[i][j] = json_obj["output_weights"][i][j]

trainer = OnlineTrainer(robot, network)

choice = ''
while choice!='y' and choice !='n':
    choice = input('Do you want to learn? (y/n) --> ')

if choice == 'y':
    trainer.training = True
elif choice == 'n':
    trainer.training = False

target = input("Enter the first target : x y --> ")
target = target.split()
for i in range(len(target)):
    target[i] = float(target[i])
print('New target : [%d, %d]'%(target[0], target[1]))

continue_running = True
while(continue_running):

    thread = threading.Thread(target=trainer.train, args=(target,))
    trainer.running = True
    thread.start()

    #Ask for stop running
    input("Press Enter to stop the current training")
    show_animation=False
    trainer.running = False
    choice = ''
    while choice!='y' and choice !='n':
        choice = input("Do you want to continue ? (y/n) --> ")

    if choice == 'y':
        choice_learning = ''
        while choice_learning != 'y' and choice_learning !='n':
            choice_learning = input('Do you want to learn ? (y/n) --> ')
        if choice_learning =='y':
            trainer.training = True
            show_animation=True
        elif choice_learning == 'n':
            trainer.training = False
            show_animation=True
        target = input("Move the robot to the initial point and enter the new target : x y radian --> ")
        target = target.split()
        for i in range(len(target)):
            target[i] = float(target[i])
        print('New target : [%d, %d]'%(target[0], target[1]))
    elif choice == 'n':
        continue_running = False


json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")