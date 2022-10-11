import random
import numpy as np
import pygame as pg
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def FormatBoard(board):
    str = np.array2string(board, precision=0, separator=' ',
                      suppress_small=True)
    
    
    str = str.replace("", " ", 1)
    str = str.replace("]", "")
    str = str.replace("[", "")
    str = str.replace(r".", '')                  
    str = str.replace(r"1", r'P'  )
    str = str.replace(r"8",  r'X'  )
    str = str.replace(r"9",  r'O' )

    return str


class Field:
    def __init__(self,size,item_pickup,item_dropoff,start_position):
        self.item_pickup = item_pickup
        self.item_dropoff = item_dropoff
        self.position = start_position
        self.size = size
        self.haveitem = False

    def initField(self):
        return np.zeros((self.size,self.size))

    def draw(self):
        self.field = self.initField()
        if self.haveitem:
            self.field[self.position[0],self.position[1]] = 8
            self.field[item_dropoff[0],item_dropoff[1]] = 9
        else:    
            self.field[self.position[0],self.position[1]] = 1
            self.field[self.item_pickup[0], self.item_pickup[1]] = 8
            self.field[item_dropoff[0],item_dropoff[1]] = 9
        board_formatted = FormatBoard(self.field)
        print(board_formatted)
        pg.time.wait(150)
        print("\033[F"*21)



    def get_number_of_states(self):
        return self.size*self.size*self.size*self.size*2

    def get_state(self):
        state = self.position[0]*self.size*self.size*self.size*2 
        state = state + self.position[1] * self.size * self.size * 2
        state = state + self.item_pickup[0] * self.size * 2
        state = state + self.item_pickup[1] * 2
        if self.haveitem:
            state = state + 1

        return state
    
    def make_action(self,action):
        (x,y) = self.position
        if action == 0:
            if y == self.size - 1:
                return -10, False
            else:
                self.position = (x,y+1) #South
                return -1, False
        elif action == 1:
            if y == 0:
                return -10,False
            else:
                self.position = (x,y-1) #North
                return -1,False
        elif action == 2:
            if x == 0:
                return -10,False
            else:
                self.position = (x-1,y) #West
                return -1,False
        elif action == 3:
            if x == self.size - 1:
                return -10,False
            else:
                self.position = (x+1,y) #East
                return -1,False
        elif action == 4:
            if self.haveitem:
                return -10, False
            elif self.item_pickup != (x,y):
                return -10,False
            else:
                self.haveitem = True
                return 20,False
        elif action == 5:
            if not self.haveitem:
                return -10,False
            elif self.item_dropoff != (x,y):
                self.item_pickup = (x,y)
                self.haveitem = False
                return -10,False
            else:
                return 20,True  

field = Field(10, (0,0), (9,9), (9,0))

def naive_solution():
    size = 10
    item_start =(0,0)
    item_dropoff = (9,9)
    start_position = (9,0)
    field = Field(size,item_start,item_dropoff,start_position)
    done = False
    steps = 0

    while not done:
        action = random.randint(0,5)
        reward , done = field.make_action(action)
        steps += 1
       
    return steps

size = 10
item_start =(0,0)
item_dropoff = (9,9)
start_position = (9,0)


number_of_states = field.get_number_of_states()
number_of_actions = 6
q_table = np.zeros((number_of_states,number_of_actions))
epsilon = 0.1
alpha = 0.01
gamma = 0.6
c_list =[]
rewardMax = 0
for _ in range(100000):
    field = Field(size,item_start,item_dropoff,start_position)
    done = False
    rewardS = 0
    while not done:
        if _ != 0 and _ % 10000 == 0:
            field.draw()
            

        state = field.get_state()
        if random.uniform(0,1)<epsilon:
            action = random.randint(0,5)
        else:
            action = np.argmax(q_table[state])

        reward, done = field.make_action(action)
        
        new_state = field.get_state()
        new_state_max = np.max(q_table[new_state])
        q_table[state,action] = (1-alpha)*q_table[state,action] + alpha*(reward + gamma*new_state_max - q_table[state,action])
        rewardS+=reward
        if rewardS>rewardMax:
            rewardMax = rewardS

print(rewardMax)
    



