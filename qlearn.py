import random
import numpy as np
import time

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
    str = str.replace(r"0",  r' ' )
    return str

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
        print("---"*21)
        print(board_formatted)
        time.sleep(0.25)

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


size = 10
item_start =(8,2)
item_dropoff = (7,7)
start_position = (1,0)

field = Field(size,item_start,item_dropoff,start_position)
number_of_states = field.get_number_of_states()
number_of_actions = 6
q_table = np.zeros((number_of_states,number_of_actions))


epsilon = 0.1
alpha = 0.01
gamma = 0.6


minSteps = 50
maxReward = 0
for epoch in range(100000):
    field = Field(size,item_start,item_dropoff,start_position)

    done = False
    drawing = False
    rewardS = 0
    steps = 0

    while not done:
        if epoch != 0 and epoch % 2000 == 0:
            field.draw()
            drawing = True

        state = field.get_state()
        if random.uniform(0,1)<epsilon and not drawing:
            action = random.randint(0,5)
        else:
            action = np.argmax(q_table[state])

        reward, done = field.make_action(action)
        
        new_state = field.get_state()
        new_state_max = np.max(q_table[new_state])
        q_table[state,action] = (1-alpha)*q_table[state,action] + alpha*(reward + gamma*new_state_max - q_table[state,action])

        steps+=1
        rewardS+=reward

    if steps<minSteps:
        minSteps = steps
        maxReward = rewardS

    if epoch != 0 and epoch % 2000 == 0:
        print(f"Min steps: {minSteps}, Max reward: {maxReward}, Epoch: {epoch}")

print(minSteps,maxReward)
    



