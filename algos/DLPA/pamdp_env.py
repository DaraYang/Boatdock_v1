##### This is the env that being called for run_boat_algo files ####################
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
import PIL.Image as Image
import gym
import random
import math
from gym import Env, spaces
import time
import torch
import os
import pandas as pd
# from torchvision.io import read_image
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pickle
import random
import sys
#import wandb
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

# constants
INDEX_FINGER = "indx_fin"
MIDDLE_FINGER = "mid_fin"
RING_FINGER = "ring_fin"

ACTION_LOOKUP = {
    0: INDEX_FINGER,
    1: MIDDLE_FINGER,
    2: RING_FINGER
}
PARAMETERS_MIN =np.array([-1, -1, -1])
PARAMETERS_MAX = np.array([1, 1, 1])

class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


    
class Startpoint(Point):
    def __init__(self,name,x_max, x_min, y_max, y_min):
        super(Startpoint,self).__init__(name,x_max, x_min, y_max, y_min)
        self.icon = 1- cv2.imread("/home/tianqiu/Dara/BoatEnv_v1/starticon.png") / 255.0
        self.icon_w = 10
        self.icon_h = 10
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class Endpoint(Point):
    def __init__(self,name,x_max, x_min, y_max, y_min):
        super(Endpoint,self).__init__(name,x_max, x_min, y_max, y_min)
        self.icon = 1- cv2.imread("/home/tianqiu/Dara/BoatEnv_v1/endicon.png") / 255.0
        self.icon_w = 10
        self.icon_h = 10
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class Boatone(Point):
    def __init__(self,name,x_max, x_min, y_max, y_min):
        super(Boatone,self).__init__(name,x_max, x_min, y_max, y_min)
        self.icon = 1- cv2.imread("/home/tianqiu/Dara/BoatEnv_v1/boat1.png") / 255.0
        self.icon_w = 10
        self.icon_h = 10
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

        
class Boattwo(Point):
    def __init__(self,name,x_max, x_min, y_max, y_min):
        super(Boattwo,self).__init__(name,x_max, x_min, y_max, y_min)
        self.icon = 1- cv2.imread("/home/tianqiu/Dara/BoatEnv_v1/boat2.png") / 255.0
        self.icon_w = 10
        self.icon_h = 10
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

        
class Boatzero(Point):
    def __init__(self,name,x_max, x_min, y_max, y_min):
        super(Boatzero,self).__init__(name,x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("/home/tianqiu/Dara/BoatEnv_v1/boat0.png") / 255.0
        self.icon_w = 10
        self.icon_h = 10
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))     

class boatEnv(Env):
    def __init__(self,sg=None,choice=None,km=None,testmode=None):
        super(boatEnv, self).__init__()
        self.alpha = 200
        self.beta = 300
        self.sg = sg
        self.choice = choice # cong or incong env
        self.km = km
        self.centerx = 640
        self.centery = 512
        # Define a iamge based observation space
        self.canvasshape = (250,250,1)
        self.canvas_space = spaces.Box(low=0, high=255, shape=(250, 250, 1), dtype=np.uint8)

        # Define a feature based observation space
        self.observation_shape = (15,)
        # self.observation_space = spaces.Box(low = -np.inf, 
        #                                     high = np.inf,
        #                                     shape = (16,),
        #                                     dtype = np.float64)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-1000, high=1000, shape=self.observation_shape, dtype=np.float32),
            spaces.Discrete(200),  # steps (200 limit is an estimate)
        ))
        # define the action space of the environment
        # action_space means press which button for how many seconds
        # e.g. actions = [0,0,2] means press the third button for 2 seconds
        # self.action_space = spaces.Box(low=-1, high=1,shape=(3,),dtype = np.float32)
        # my action space has 3 actions, left, middle or right button. 
        num_actions = len(ACTION_LOOKUP)
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),  # actions
            spaces.Tuple(  # parameters
                tuple(spaces.Box(np.array([PARAMETERS_MIN[i]]), np.array([PARAMETERS_MAX[i]]), dtype=np.float32) for i in range(num_actions))
            )
        ))        
        # Create a canvas to render the environment images upon 
        self.canvas = np.ones(self.canvasshape) * 1
        
        # Define elements present inside the environment
        self.elements = []
        self.max_fuel = 5
        self.time_limit = 10
        self.fuel_left = 5
        # Permissible area of boat to be 
        self.y_min = 512-77
        self.x_min = 640-77
        self.y_max = 512+77
        self.x_max = 640+77
        self.sgdist = 0
        self.boatx,self.boaty = 0,0
        self.currv = [0,0]
        self.testmode = testmode
        self.disparam = 1
        self.ifi = 0.02
        self.startx,self.starty,self.endx,self.endy=0,0,0,0
    def draw_elements_on_canvas(self):
        # Init the canvas 
        self.canvas = np.ones(self.canvasshape) * 1

        # Draw the element on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = int(elem.x-520), int(elem.y-392)
            self.canvas[x:x + elem_shape[1], y:y + elem_shape[0]] = elem.icon
    
    def optimized_boat(self):
        #compute offset angle and return the optimal boat for this env(0 is cong, 1 is incong)
        offset = 0

        sx,sy,ex,ey = self.elements[0].x,self.elements[0].y,self.elements[1].x,self.elements[1].y
        sg = [[sx,sy],[ex,ey]]
        dist = math.sqrt((sg[1][1] - sg[0][1])**2 + (sg[1][0] - sg[0][0])**2)
        sgangle = [sg[1][i] - sg[0][i] for i in range(len(sg[0]))]
        if dist == 0.0:
            return -1
        else:
            sgv = [sgangle[i]/dist for i in range(len(sgangle))]
            boatdir = [[0,1], [-math.sqrt(3)/2, -1/2],[math.sqrt(3)/2,-1/2], [0,-1],[-math.sqrt(3)/2, 1/2], [math.sqrt(3)/2, 1/2]]
            offsettoboat = []
            for i in range(6):
                offsettoboat.append(math.degrees(math.acos(sgv[0]*boatdir[i][0] + sgv[1]*boatdir[i][1])))
            offset = min(offsettoboat[0:3])

            return 0 if offset>30 else 1
        
    def custom_canvas(self,custom_start,custom_end):
        # input: start and end point
        # output: optimal boat in this senario
        self.elements[0].x,self.elements[0].y = custom_start[0],custom_start[1]
        self.elements[1].x,self.elements[1].y = custom_end[0],custom_end[1]
        return self.optimized_boat()
    
    def initsg(self):
        if self.sg is not None:
            startx,starty = self.sg[0][0],self.sg[0][1]
            endx,endy = self.sg[1][0],self.sg[1][1]
            self.startpoint = Startpoint("startpoint", self.x_max, self.x_min, self.y_max, self.y_min)
            self.startpoint.set_position(startx,starty)
            self.endpoint = Endpoint("endpoint", self.x_max, self.x_min, self.y_max, self.y_min)
            self.endpoint.set_position(endx,endy)
            self.sgdist = self.comp_dist(startx,starty,endx,endy)
        else:
            self.ideal_boat=None
            while self.sgdist < 100 or self.ideal_boat != self.testmode:
                startx = random.randrange(self.x_min, self.x_max)
                starty = random.randrange(self.y_min, self.y_max)
                endx = random.randrange(self.x_min, self.x_max)
                endy = random.randrange(self.y_min, self.y_max)
                self.sgdist = self.comp_dist(startx,starty,endx,endy)
                self.startpoint = Startpoint("startpoint", self.x_max, self.x_min, self.y_max, self.y_min)
                self.startpoint.set_position(startx,starty)
                self.endpoint = Endpoint("endpoint", self.x_max, self.x_min, self.y_max, self.y_min)
                self.endpoint.set_position(endx,endy)
                self.elements = [self.startpoint,self.endpoint]
                self.ideal_boat = self.optimized_boat() if self.testmode is not None else None
        self.sg = [[startx,starty],[endx,endy]]
        self.startx, self.starty, self.endx, self.endy = startx,starty,endx,endy
   
    def reset(self):
        # init and reset 
        self.fuel_left = self.max_fuel
        self.ep_return  = 0
        self.ideal_boat = -1
        self.time_limit = 1000

        self.initsg()

        self.disparam = self.sgdist/140

        # Initilize boat
        boatx = self.startx
        boaty = self.starty
        if self.choice is None:
            self.choice = random.randint(0, 1)
        if self.choice == 1:
            self.boatone = Boatone("boatone", self.x_max, self.x_min, self.y_max, self.y_min)
            self.boatone.set_position(boatx,boaty)
        elif self.choice == 0:
            self.boattwo = Boattwo("boattwo", self.x_max, self.x_min, self.y_max, self.y_min)
            self.boattwo.set_position(boatx,boaty)
            
        self.boatx,self.boaty = boatx,boaty
        # Intialise the elements
        # self.elements = [self.startpoint,self.boatzero,self.endpoint]
        if self.choice == 1:
            self.elements = [self.startpoint,self.endpoint,self.boatone]
            if not self.km:
                self.km=-1
        else:
            self.elements = [self.startpoint,self.endpoint,self.boattwo]
            if not self.km:
                self.km = random.randint(0, 1)
        # optimal boat
        self.op_choice = self.optimized_boat()
        # Reset the Canvas 
        self.canvas = np.ones(self.canvasshape) * 1
        self.chooseboat(self.choice,self.km)
        self.dist = self.comp_dist(self.startx,self.starty,self.endx,self.endy)
        # Draw elements on the canvas
        # self.draw_elements_on_canvas()
        self.observation_space = self.get_obs()

        # return the observation
        return self.observation_space, 0
    
   
    def customcanvas(self,customs,custome):
        self.startpoint = Startpoint("startpoint", self.x_max, self.x_min, self.y_max, self.y_min)
        self.startpoint.set_position(customs[0],customs[1])
        self.endpoint = Endpoint("endpoint", self.x_max, self.x_min, self.y_max, self.y_min)
        self.endpoint.set_position(custome[0],custome[1])
        self.elements = [self.startpoint,self.endpoint]
        self.op_choice = self.optimized_boat()
        # Reset the Canvas 
        self.canvas = np.ones(self.canvasshape) * 1
        self.fuel_left = 5
        # Draw elements on the canvas
        # self.draw_elements_on_canvas()
        self.observation_space = self.get_obs()

        # return the observation
        return self.canvas
        

    def render(self,inputstates,inputactions,fname='testname.png'):
        # This function takes in states and actions and save the rendered image
        inputstate = inputstates[0]
        inputaction = inputactions
        sx,sy,ex,ey= inputstate[0],inputstate[1],inputstate[2],inputstate[3]
        self.startpoint = Startpoint("startpoint", self.x_max, self.x_min, self.y_max, self.y_min)
        self.endpoint = Endpoint("endpoint", self.x_max, self.x_min, self.y_max, self.y_min)
        self.startpoint.set_position(sx,sy)
        self.endpoint.set_position(ex,ey)

        self.draw_elements_on_canvas()
        locs = []
        locs.append[[self.startx,self.starty]]
        for a in inputaction:
            _, _, _, _ = self.step(a)
            newboatxy = self.observation_space[9:11]
            locs.append(newboatxy)
        plt.figure(figsize=(8, 6))

        # Plot the trajectory
        x_values = [pair[0] for pair in locs]
        y_values = [pair[1] for pair in locs]
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Trajectory')

        # Plot the stationary points
        plt.scatter(self.startx, self.starty, color='g', label='start', zorder=5)
        plt.scatter(self.endx, self.endy, color='r', label='end', zorder=5)

        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('Trajectory produced by DT')
        plt.legend()

        # Show plot
        plt.savefig(fname)
        plt.close()
                
    def close(self):
        cv2.destroyAllWindows()
        
    def get_action_meanings(self):
        return {0:"left", 1:"middle", 2:"right", 3:"do nothing"}


    # This function dectect if two item met each other
    def has_collided(self, elem1, elem2):
        
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False
    
    
    def comp_dist(self,x1,y1,x2,y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) 
        
    def denorm_action(self, normaction):
    # this function denormalize the actions from -1 to 1 to 0 to 5
        denorm_action = (normaction + 1) / 2 * 5
        return denorm_action
    
    def step(self, action):
        # action is a vector of [a1,a2,a3] being pressing ith button for ai seconds
        # Flag that marks the termination of an episode
        done = False
        act_indx = action[0]
        act = ACTION_LOOKUP[act_indx]
        param = action[1][act_indx]
        param = np.clip(param, PARAMETERS_MIN[act_indx], PARAMETERS_MAX[act_indx])
        # if not self.action_space.contains(action):
        #     return self.observation_space, 0, done, {}

        # else:
        # denormalize action
        denorm_action = self.denorm_action(param)
        denorm_action_vec = np.zeros((1,3))
        denorm_action_vec[0][act_indx] = denorm_action
        self.fuel_left -= param.item()  
        beta = 0           
        boatdir = np.array([[z * 10 for z in y] for y in self.dirs])
        denorm_action_vec = np.array(denorm_action_vec)/self.ifi
        self.currv = denorm_action_vec @ boatdir * self.ifi
        self.currv = self.currv[0]
        pv = np.array([-0.6667,10,0])
        D1 = np.array([(pv[0]*denorm_action_vec[i]**3 + pv[1]*denorm_action_vec[i]**2 + pv[2])*self.ifi for i in range(len(denorm_action_vec))])
        D2 = [[D1*x for x in y] for y in boatdir]
        D2 = D1[:,np.newaxis][0]@boatdir
        displacement = np.sum(D2,axis=0)
        if self.choice == 1:
            self.boatone.move(displacement[0],displacement[1])
        else:
            self.boattwo.move(displacement[0],displacement[1])
        # self.boatx,self.boaty = bx,by
        
        for elem in self.elements:
            if isinstance(elem, Boatone):
                bx = elem.get_position()[0]
                by = elem.get_position()[1]
                currdist = self.comp_dist(self.endx,self.endy,bx,by)  
                # If the boat reached the boarder, end the episode, remove the boat and end point from canvas
                if elem.get_position()[0] <= self.x_min or elem.get_position()[0] >= self.x_max \
                or elem.get_position()[1] <= self.y_min or elem.get_position()[1] >= self.y_max:
                    done = True
                    self.ep_return = 0
                    self.observation_space = self.get_obs() 
                    obs = (self.observation_space,1)
                    return obs, self.ep_return, done, {}
                    #self.elements.remove(self.boatzero)
                if self.has_collided(self.endpoint, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    # print('yayyyyyyyy')
                    # self.ep_return = (self.fuel_left/self.max_fuel)*self.disparam*self.beta
                    # self.ep_return=self.alpha*(1/currdist) - 2 * ((self.max_fuel-self.fuel_left)/self.max_fuel)
                    # self.ep_return = self.alpha*(1/currdist)
                    self.ep_return = (self.fuel_left/self.max_fuel)*10
                    self.observation_space = self.get_obs() 
                    obs = (self.observation_space,1)
                    return obs, self.ep_return, done, {}
                    #self.elements.remove(self.boatzero)

            elif isinstance(elem, Boattwo):
                bx = elem.get_position()[0]
                by = elem.get_position()[1]
                currdist = self.comp_dist(self.endx,self.endy,bx,by)  
                # If the boat reached the boarder, end the episode, remove the boat and end point from canvas
                if elem.get_position()[0] <= self.x_min or elem.get_position()[0] >= self.x_max \
                or elem.get_position()[1] <= self.y_min or elem.get_position()[1] >= self.y_max:
                    done = True
                    self.ep_return = 0
                    self.observation_space = self.get_obs() 
                    obs = (self.observation_space,1)
                    return obs, self.ep_return, done, {}
            
                # If the boat has reached the endpoint.
                if self.has_collided(self.endpoint, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    # print('yayyyyyyyy')
                    # self.ep_return = (self.fuel_left/self.max_fuel)*self.disparam*self.beta
                    # self.ep_return=self.alpha*(1/currdist) - 2 * ((self.max_fuel-self.fuel_left)/self.max_fuel)
                    # self.ep_return = self.alpha*(1/currdist)
                    self.ep_return = (self.fuel_left/self.max_fuel)*10
                    self.observation_space = self.get_obs() 
                    obs = (self.observation_space,1)
                    return obs, self.ep_return, done, {}
                    #self.elements.remove(self.boatzero)

        self.boatx,self.boaty = bx,by
        # Increment the episodic return 
        # currdist = self.comp_dist(self.endx,self.endy,bx,by)           
        #currdist = math.sqrt((self.endpoint.x_min - bx)**2 + (self.endpoint.y_min - by)**2) 
        if currdist < 1e-5:
            currdist = 1
        self.ep_return = self.alpha*(1/currdist) - 2 * ((self.max_fuel-self.fuel_left)/self.max_fuel)
        
        # Draw elements on the canvas
        #self.draw_elements_on_canvas()

        # If out of fuel, end the episode.
        if self.fuel_left <= 0 or self.time_limit <= 0 or self.outofboundary():
            done = True
            self.ep_return = 0
        if done == False:
            self.ep_return = self.alpha*(1/currdist)
            self.time_limit -= 1
        self.dist = self.comp_dist(bx,by,self.endx,self.endy)
        self.observation_space = self.get_obs() 
        # print(self.ep_return)
        obs = (self.observation_space,1)
        return obs, self.ep_return, done, {}
    def get_obs(self):
        # currx,curry,sx,sy,ex,ey,cx,cy,r,v1,v2,v3,
        return np.concatenate((
            
            np.array([self.startx, self.starty,
            self.endx,self.endy]),
            np.array([640,512]),
            np.array([120]),
            np.array([self.choice,
            self.km,
            self.boatx,self.boaty]),
            self.currv,
            np.array([self.fuel_left/5]),
            np.array([self.dist/self.comp_dist(self.startx,self.starty,self.endx,self.endy)]))
        )
    def outofboundary(self):
        return self.comp_dist(self.boatx,self.boaty,self.centerx,self.centery) > 125
    

    def chooseboat(self,choice,choicekm):
        self.choice = choice
        self.km = choicekm
        if self.choice == 1:
            self.dirs = [[-0.866,0.5],[0,-1],[0.866,0.5]]
        else:
            if self.km is None:
                self.km = random.randint(0,1)
            else:
                if self.km == 1:
                    self.dirs = [[0.866,-0.5],[-0.866,-0.5],[0,1]]
                else:
                    self.dirs = [[0,1],[0.866,-0.5],[-0.866,-0.5]]