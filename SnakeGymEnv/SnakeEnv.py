import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import gym
from gym import spaces

# import cv2
# import random
# import time

from collections import deque


def viz_game(snake_position, apple_position, snake_head, head_direction):
    pass
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax = plt.gca()
#     ax.set_facecolor('#FFF9C4')

#     plt.xlim([0, 150])
#     plt.ylim([0, 150])

#     plt.xticks(np.arange(0, 151, 10))
#     plt.yticks(np.arange(0, 151, 10))


#     ax.grid(True, linewidth=0.5, color='gray', linestyle='-')
#     ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

#     # Plotting Apple
#     img = plt.imread("SnakeGymEnv/Graphics/apple.png")
    
#     snakehead1 = plt.imread("SnakeGymEnv/Graphics/head_right.png")
#     snakehead3 = plt.imread("SnakeGymEnv/Graphics/head_down.png")
#     snakehead0 = plt.imread("SnakeGymEnv/Graphics/head_left.png")
#     snakehead2 = plt.imread("SnakeGymEnv/Graphics/head_up.png")
    
#     snakebodyimg = plt.imread("SnakeGymEnv/Graphics/body_horizontal.png")
    
#     snaketail1 = plt.imread("SnakeGymEnv/Graphics/tail_right.png")
#     snaketail3 = plt.imread("SnakeGymEnv/Graphics/tail_down.png")
#     snaketail0 = plt.imread("SnakeGymEnv/Graphics/tail_left.png")
#     snaketail2 = plt.imread("SnakeGymEnv/Graphics/tail_up.png")
    
# #     for apple_position in apple_position_list:
# #         ax.imshow(img, extent=[apple_position[0], apple_position[0]+10, apple_position[1], apple_position[1]+10])
#     ax.imshow(img, extent=[apple_position[0], apple_position[0]+10, apple_position[1], apple_position[1]+10])

    
#     if head_direction == 0:
#         ax.imshow(snakehead0, extent=[snake_head[0], snake_head[0]+10, snake_head[1], snake_head[1]+10])
#         for position in snake_position[1:]:
#             ax.imshow(snakebodyimg, extent=[position[0], position[0]+10, position[1], position[1]+10])
        
#     elif head_direction == 1:
#         ax.imshow(snakehead1, extent=[snake_head[0], snake_head[0]+10, snake_head[1], snake_head[1]+10])
#         for position in snake_position[1:]:
#             ax.imshow(snakebodyimg, extent=[position[0], position[0]+10, position[1], position[1]+10])

#     elif head_direction == 2:
#         ax.imshow(snakehead2, extent=[snake_head[0], snake_head[0]+10, snake_head[1], snake_head[1]+10])
#         for position in snake_position[1:]:
#             ax.imshow(snakebodyimg, extent=[position[0], position[0]+10, position[1], position[1]+10])

#     elif head_direction == 3:
#         ax.imshow(snakehead3, extent=[snake_head[0], snake_head[0]+10, snake_head[1], snake_head[1]+10])
#         for position in snake_position[1:]:
#             ax.imshow(snakebodyimg, extent=[position[0], position[0]+10, position[1], position[1]+10])

#     plt.draw()
#     plt.pause(1)
#     # fig = plt.gcf()
#     # plt.savefig("frame_%d.png")
#     # fig.canvas.draw()
#     # image = Image.frombytes('RGB', fig.canvas.get_width_height(), 
#                         # fig.canvas.tostring_rgb(), 
#                         # decoder_name='raw')



def viz_game_over(apple_count):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plt.gca()
    plt.xlim([0, 500])
    plt.ylim([0, 500])

    font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 16}
    ax.text(180, 250, 'Your Apple Count is {}'.format(apple_count), fontdict=font)

    plt.title("Game Over!", fontdict=font)
    plt.show()
    
    
def viz_game_win(apple_count):
    pass
    

def boundary_collision(snake_head):
     # When snake dimensions go beyond the grid dimensions
    if snake_head[0]>=150 or snake_head[0]<0 or snake_head[1]>=150 or snake_head[1]<0 :
        return 1 # True
    else:
        return 0 # False

    
def self_collision(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

# MAX_SNAKE_LEN = 15

class Snake(gym.Env):

    def __init__(self, render=False, MAX_SNAKE_LEN=15):
        super(Snake, self,).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=-500, high=500, shape=(5+20,), dtype=np.float32)
        self.show_env = render
        self.timestep = 0
        self.timestep_max = 4000
        self.head_direction = 0
        self.max_snake_len = MAX_SNAKE_LEN

    def step(self, action):
        self.prev_actions.append(action)
        self.timestep += 1

#         print("Snake Direction Initial: ",self.head_direction)
#         print("Snake Head Initial: ",self.snake_head)
        
        if action == 0:
            self.head_direction = (self.head_direction + 1) % 4
        elif action == 1:
            self.head_direction = (self.head_direction - 1) % 4
        elif action == 2:
            pass
        
        if self.head_direction == 0:
            self.snake_head[0] += 10
        elif self.head_direction == 1:
            self.snake_head[1] += 10
        elif self.head_direction == 2:
            self.snake_head[0] -= 10
        elif self.head_direction == 3:
            self.snake_head[1] -= 10
            
#         print("Snake Direction After: ",self.head_direction)
#         print("Snake Head After: ",self.snake_head)

        # Increase Snake length on eating apple
        apple_reward = 0
        if self.snake_head == self.apple_position:
            self.apple_count += 1
            self.apple_position = self.apple_position_list[self.apple_count]
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 500
            if len(self.snake_position) == self.max_snake_len:
                self.reward += 500

            
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
            
        
        if self.show_env:
            viz_game(self.snake_position, self.apple_position, self.snake_head, self.head_direction)
        
        
        # On collision end episode and print the score
        if boundary_collision(self.snake_head) == 1 or self_collision(self.snake_position) == 1:
            if self.show_env:
                viz_game_over(self.apple_count)
            self.done = True
            self.reward = -100


        # REWARD DYNAMICS
        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        self.reward =  (150 - euclidean_dist_to_apple)/10 + apple_reward

        ## if snake dies
#         if self.done:
#             self.reward = -100                                     
        info = {}
    
        if self.timestep == self.timestep_max:
            self.done = True

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, info

    def reset(self):
        
        # Initial Snake and Apple position
        self.snake_position = [[70,80],[60,80],[50,80]]
        self.apple_count = 0
        self.apple_position_list = [[110,60],[50,40],[50,80],[60,110],[70,110],[80,90],[110,60],\
                                    [40,50],[40,80],[40,110],[120,20],[120, 120],[30,120],[90,40],[30,20]]
        self.apple_position = self.apple_position_list[self.apple_count]
        
        self.prev_head_direction = 1
        self.head_direction = 0
        self.snake_head = [70,80]

        self.prev_reward = 0
        self.timestep = 0

        self.done = False                                                                    

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen = 20)
        for i in range(20):
            self.prev_actions.append(-1) 

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation

