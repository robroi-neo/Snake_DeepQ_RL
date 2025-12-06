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
SNAKE_FOOD = (34, 34, 34)
BONUS_FOOD = (214, 52, 52)
SNAKE_BODY = (34, 34, 34)
SNAKE_BODY_SHADOW = (156, 174, 142) 
BACKGROUND = (170,204,153)

BLOCK_SIZE = 20
FPS = 60

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

        self.move_delay = 120  # milliseconds per movement step (≈8 tiles/sec)
        self.last_move_time = 0


        self.clock = pygame.time.Clock()
        self.reset()
        
    
    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w // 2, self.h // 2)

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
        
        # 2. move only if enough time passed
        current_time = pygame.time.get_ticks()

        moved = False
        if current_time - self.last_move_time > self.move_delay:
            self.last_move_time = current_time
            self._move(self.direction)
            self.snake.insert(0, self.head)
            moved = True

        if moved:
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
                if self.bonus_counter == 4:
                    self._place_bonus()
                    self.bonus_counter = 0

                self._place_food()

            elif self.bonus is not None and self.head == self.bonus:
                self.score += 10
                self.bonus = None
                self.bonus_spawn_time = None

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
        self.clock.tick(FPS)
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
