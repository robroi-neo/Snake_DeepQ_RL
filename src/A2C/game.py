import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np 

# PLEASE DON'T TOUCH THIS FILE PARA WALAY CONFILCT PLES KAPOY MAG RESOLVE UG MERGE CONFLICS HUHUHUHU MALUOY KA
# MAG EDIT PAKOG REWARD SHAPING DIRI


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
SNAKE_FOOD = (34, 34, 34)
BONUS_FOOD = (214, 52, 52)
SNAKE_BODY = (34, 34, 34)
SNAKE_BODY_SHADOW = (156, 174, 142) 
BACKGROUND = (170,204,153)

BLOCK_SIZE = 40

# this is the self's frame rate
# you may increase this value for faster training speed

class SnakeGameAI:

    def __init__(self, w=8, h=10, fps=2000):
        self.w = w * 40
        self.h = h * 40
        self.BlockSize = BLOCK_SIZE
        self.fps = fps
        
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
        self.head = Point(self.w // 2, self.h // 2)
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
        
        # 2. move only if enough time passed
        self._move(action)
        self.snake.insert(0, self.head)

        # --- distance-based reward shaping ---
        # compute old & new food distance
        old_distance = abs(self.snake[1].x - self.food.x) + abs(self.snake[1].y - self.food.y)
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        reward = 0
        game_over = False

        # 3. Check if game is over
        if self.is_collision() or self.frame_iteration > 30 * len(self.snake):
            game_over = True
            reward = -50
            return reward, game_over, self.score

        # distance shaping BEFORE other rewards
        if new_distance < old_distance:
            reward += 10     # moved closer
        elif new_distance > old_distance:
            reward -= 1      # moved farther

        # 4. Eating logic
        if self.head == self.food:
            self.score += 1
            reward += 50      # keep original
            self.bonus_counter += 1

            if self.bonus_counter == 8:
                self._place_bonus()
                self.bonus_counter = 0

            self._place_food()

        elif self.bonus is not None and self.head == self.bonus:
            self.score += 10
            reward += 150
            self.bonus = None
            self.bonus_spawn_time = None

        else:
            self.snake.pop()


        # 5. bonus timer expiration
        if self.bonus is not None:
            now = pygame.time.get_ticks()
            # print(now - self.bonus_spawn_time)
            # make this scale with game speed
            BASE_BONUS_TIME = 4000
            bonus_lifetime = BASE_BONUS_TIME * (20 / self.fps)
            if now - self.bonus_spawn_time >= bonus_lifetime:  # 4 seconds
                self.bonus = None
                self.bonus_spawn_time = None

        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(self.fps)
        
        # return
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """check if snake is outside the map"""
        if pt is None:
            pt = self.head
        
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def is_self_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BACKGROUND)

        # shadow is used for food, snake, and bonus
        SHADOW_OFFSET_X = 0
        SHADOW_OFFSET_Y = 4
        # shadow for food
        pygame.draw.rect(
            self.display,
            SNAKE_BODY_SHADOW,  # you can use a slightly different color if you want
            (self.food.x + SHADOW_OFFSET_X, self.food.y + SHADOW_OFFSET_Y, BLOCK_SIZE, BLOCK_SIZE),
            border_radius=4
        )

        # normal food
        pygame.draw.rect(
            self.display,
            SNAKE_FOOD,
            (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
            border_radius=4
        )
        # bonus food
        if self.bonus:
            # shadow for bonus
            pygame.draw.rect(
                self.display,
                SNAKE_BODY_SHADOW,  # shadow color (same as normal food shadow)
                (self.bonus.x + SHADOW_OFFSET_X, self.bonus.y + SHADOW_OFFSET_Y, BLOCK_SIZE, BLOCK_SIZE),
                border_radius=4
            )

            # actual bonus food
            pygame.draw.rect(
                self.display,
                BONUS_FOOD,  # your bonus color
                (self.bonus.x, self.bonus.y, BLOCK_SIZE, BLOCK_SIZE),
                border_radius=4
            )
        
        # draw snake
        for i, pt in enumerate(self.snake):
            if i == 0:  # head is always visible
                draw_shadow = True
            else:
                # if previous segment is offset exactly on top of where the shadow would be, skip
                prev = self.snake[i-1]
                shadow_rect = pygame.Rect(pt.x + SHADOW_OFFSET_X, pt.y + SHADOW_OFFSET_Y, BLOCK_SIZE, BLOCK_SIZE)
                body_rect = pygame.Rect(prev.x, prev.y, BLOCK_SIZE, BLOCK_SIZE)
                draw_shadow = not shadow_rect.colliderect(body_rect)
            
            if draw_shadow:
                pygame.draw.rect(
                    self.display,
                    SNAKE_BODY_SHADOW,
                    (pt.x + SHADOW_OFFSET_X, pt.y + SHADOW_OFFSET_Y, BLOCK_SIZE, BLOCK_SIZE),
                    border_radius=4
                )
            
            pygame.draw.rect(
                self.display,
                SNAKE_BODY,
                (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
                border_radius=4
            )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

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

    def get_state(self):
        bs = BLOCK_SIZE
        head = self.snake[0]
        point_l = Point(head.x - bs, head.y)
        point_r = Point(head.x + bs, head.y)
        point_u = Point(head.x, head.y - bs)
        point_d = Point(head.x, head.y + bs)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # normalize
        bonus_time_norm = (8 - self.bonus_counter) / 8

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Me sa harap
            (dir_r and self.is_self_collision(point_r)) or 
            (dir_l and self.is_self_collision(point_l)) or 
            (dir_u and self.is_self_collision(point_u)) or 
            (dir_d and self.is_self_collision(point_d)),

            # Me sa right
            (dir_u and self.is_self_collision(point_r)) or 
            (dir_d and self.is_self_collision(point_l)) or 
            (dir_l and self.is_self_collision(point_u)) or 
            (dir_r and self.is_self_collision(point_d)),

            # Me sa left
            (dir_d and self.is_self_collision(point_r)) or 
            (dir_u and self.is_self_collision(point_l)) or 
            (dir_r and self.is_self_collision(point_u)) or 
            (dir_l and self.is_self_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y,  # food down

            # Bonus existence
            bonus_time_norm,

            # Bonus Food Location (only if exists)
            0 if self.bonus is None else self.bonus.x < self.head.x,
            0 if self.bonus is None else self.bonus.x > self.head.x,
            0 if self.bonus is None else self.bonus.y < self.head.y,
            0 if self.bonus is None else self.bonus.y > self.head.y
        ]
        return np.array(state, dtype=float)
