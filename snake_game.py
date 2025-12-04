import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    
    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
        ]
        
        self.score = 0

        self.bonus = None
        self.bonus_spawn_time = None     
        self.bonus_counter = 0            

        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

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

    def play_step(self):
        # 1. input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # 3. collision
        if self._is_collision():
            return True, self.score
            
        # 4. eating logic
        if self.head == self.food:
            print("bonus counter: ", self.bonus_counter)
            self.score += 1

            # only normal food increments bonus counter
            self.bonus_counter += 1

            # every 8th food spawns bonus
            if self.bonus_counter == 8:
                self._place_bonus()
                self.bonus_counter = 0

            self._place_food()

        elif self.bonus is not None and self.head == self.bonus:
            self.score += 10
            self.bonus = None
            self.bonus_spawn_time = None
            self._place_food()

        else:
            self.snake.pop()

        # 5. bonus timer expiration
        if self.bonus is not None:
            now = pygame.time.get_ticks()
            # print(now - self.bonus_spawn_time)
            if now - self.bonus_spawn_time >= 4000:  # 4 seconds
                self.bonus = None
                self.bonus_spawn_time = None

        # 6. ui + clock
        self._update_ui()
        self.clock.tick(SPEED)
        return False, self.score
    
    def _is_collision(self):
        if (
            self.head.x > self.w - BLOCK_SIZE or
            self.head.x < 0 or
            self.head.y > self.h - BLOCK_SIZE or
            self.head.y < 0
        ):
            return True

        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, (pt.x+4, pt.y+4, 12, 12))
            
        # draw normal food
        pygame.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw bonus if exists
        if self.bonus:
            pygame.draw.rect(self.display, GREEN, (self.bonus.x, self.bonus.y, BLOCK_SIZE, BLOCK_SIZE))

        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGame()
    
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
        
    print('Final Score', score)
    pygame.quit()
