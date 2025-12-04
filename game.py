import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np 

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 4000 

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """ The reset function will reset the snake length, score, frame_iteration back to 0. It will also randomly spawn a new food in the map."""
        # reset game state
        self.direction = Direction.RIGHT

        # a snake is represented as a list of coordinates.
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0

        self.bonus = None
        self.bonus_spawn_time = None     
        self.bonus_counter = 0  

        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """ Randomly Place food in the map """
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)

        # check if the coordinate of the food is in the coordinate of the snake, if yes create new food.
        if self.food in self.snake:
            self._place_food()

    def _place_bonus(self):
        # only place if no bonus exists
        if self.bonus is not None:
            return
        
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        
        self.bonus = Point(x, y)
        self.bonus_spawn_time = pygame.time.get_ticks()   
        
        if self.bonus in self.snake:
            self.bonus = None
            self._place_bonus()

    def play_step(self, action):
        """ Plays a step according to the model's predicted action """
        # every step increases frame iteration by 1. this will be used to calculate how long the game is running
        self.frame_iteration += 1
        
        # 1. collect user input -> exits the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action) # move based on the action received from the model
        self.snake.insert(0, self.head) # inserts the new head position to the front of the snake body list.

        # 3. Check if game is over
        reward = 0
        game_over = False
        # game over is triggered happens when the snake is in collision or the game is taking too long
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Eating Logic
        if self.head == self.food:
            self.score += 1

            #reward of 1 for eating normal food
            reward = 10

            # only normal food increments bonus counter
            self.bonus_counter += 1

            # every 8th food spawns bonus
            if self.bonus_counter == 8:
                self._place_bonus()
                self.bonus_counter = 0

            self._place_food()

        elif self.bonus is not None and self.head == self.bonus:
            self.score += 10
            reward = 100
            self.bonus = None
            self.bonus_spawn_time = None
            self._place_food()

        else:
            # this is moving forward because i remove the last tail...
            self.snake.pop()

        # 5. bonus timer expiration
        if self.bonus is not None:
            now = pygame.time.get_ticks()
            # print(now - self.bonus_spawn_time)
            if now - self.bonus_spawn_time >= 4000:  # 4 seconds
                self.bonus = None
                self.bonus_spawn_time = None

        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # return
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """check if snake is outside the map or eats himself wink"""
        if pt is None:
            pt = self.head
        
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # draw normal food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # draw bonus if exists
        if self.bonus:
            pygame.draw.rect(self.display, GREEN, (self.bonus.x, self.bonus.y, BLOCK_SIZE, BLOCK_SIZE))

        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()

    def _move(self, action):
        # action -> [straing, right, left]
        # direction: right, left, up, down

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)