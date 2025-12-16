import pygame
import pygame.freetype
from screen_base import ScreenBase
import torch
from rect import RectWithText
import sys
import os
from game import SnakeGameAI
pygame.init()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DQN')))

from model import Linear_QNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))
from run_dqn_game import load_model as load_dqn
from run_duel_game import load_model as load_duel
clock = pygame.time.Clock()

DQN_PATH = "model/DQN/checkpoint_300.pth"
DUEL_PATH = "model/DuelDQN/checkpoint_2500.pth"


versions = [1,20,400,1000]
speeds = [1,2,4,8,16,]
dqn_versions = [
    "model/DQN/checkpoint_100.pth",
    "model/DQN/checkpoint_400.pth",         
     "model/DQN/checkpoint_1000.pth",  
     "model/DQN/checkpoint_2500.pth",  
]
duel_dqn_versions = [
    "model/DuelDQN/checkpoint_100.pth",
    "model/DuelDQN/checkpoint_400.pth",         
     "model/DuelDQN/checkpoint_1000.pth",  
     "model/DuelDQN/checkpoint_2500.pth",  
]


fps = [20,40,70,160,320]

class GameScreen(ScreenBase):
    def __init__(self, game,screen):
        super().__init__(game)
        self.font = pygame.freetype.Font('src/GameMenu/assets/menu/PressStart2p-Regular.ttf', 50)
        self.play_img = pygame.image.load('src/GameMenu/assets/game/play.png').convert_alpha()
        self.reset_img = pygame.image.load('src/GameMenu/assets/game/reset.png').convert_alpha()
        self.version_img = pygame.image.load('src/GameMenu/assets/game/dropdown.png').convert_alpha()
        self.speed_img = pygame.image.load('src/GameMenu/assets/game/dropdown.png').convert_alpha()
        self.play_rect = self.play_img.get_rect(topleft=(32, 203))
        self.reset_rect = self.reset_img.get_rect(topleft=(32, 278))
        self.version_rect = self.version_img.get_rect(topleft=(32,437))
        self.version_buttons = []
        self.selected_version = versions[0]
        self.version_isActive = False
        self.speed_rect = self.speed_img.get_rect(topleft=(32,596))
        self.speed_buttons = []
        self.selected_speed = speeds[0]
        self.speed_isActive = False
        self.fps = 20,
        self.isPlaying = False
        for index, v in enumerate(versions):
            version_button = RectWithText(
                32,
                483 + (index * 46),
                150,
                50,
                "v" + str(v),
                14,
                fill_color=(170,204,153),
                text_color=(32,32,32),
                border_width=4,
                border_color=(32,32,32),
                center=False
            )
            self.version_buttons.append((version_button, v))
        for index, speed in enumerate(speeds):
            speed_button = RectWithText(
                32,
                550 - (index * 46),
                150,
                50,
                str(speed) + "x",
                14,
                fill_color=(170,204,153),
                text_color=(32,32,32),
                border_width=4,
                border_color=(32,32,32),
                center=False
            )
            self.speed_buttons.append((speed_button, speed))
        
        # FIX: Use dimensions that are multiples of BLOCK_SIZE (40)
        # 360 = 9 blocks, 680 = 17 blocks
        snake_container_rect = pygame.Rect(214, 110, 360, 600)
        self.snake_surface1 = screen.subsurface(snake_container_rect)
        snake_container_rect2 = pygame.Rect(609, 110, 360, 600)
        self.snake_surface2 = screen.subsurface(snake_container_rect2)
        self.agent1 = load_dqn(dqn_versions[0])
        self.score1 = 0
        
        self.agent2 = load_duel(duel_dqn_versions[0])
        self.score2 = 0

        # FIX: Pass width and height in pixels that are multiples of 40
        self.game1 = SnakeGameAI(w=360, h=600, fps=10, surface=self.snake_surface1)
        self.game1.game_over = True
        self.game2 = SnakeGameAI(w=360, h=600, fps=10, surface=self.snake_surface2)
        self.game2.game_over = True
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.game.change_screen("game") 

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # left click
          if self.play_rect.collidepoint(event.pos):
                self.isPlaying = True
                self.game1.game_over = False
                self.game2.game_over = False
                self.score1 = 0
                self.score2 = 0
                self.game1.reset()
                self.game2.reset()
          if self.reset_rect.collidepoint(event.pos):
                self.game1.reset()
                self.game2.reset()
                self.game1.game_over = True
                self.game2.game_over = True
                self.isPlaying = False
                self.score1 = 0
                self.score2 = 0
          if self.version_rect.collidepoint(event.pos) and not self.speed_isActive:
                self.version_isActive = not self.version_isActive
          if self.speed_rect.collidepoint(event.pos) and not self.version_isActive:
                self.speed_isActive = not self.speed_isActive
          for index, (version_button, v) in enumerate(self.version_buttons):
            if  version_button.rect.collidepoint(event.pos) and self.version_isActive:
              self.selected_version = versions[index]
              self.version_isActive = False
              self.agent1 = load_dqn(dqn_versions[index])  
              self.agent2 = load_duel(duel_dqn_versions[index])
          for index, (speed_button, v) in enumerate(self.speed_buttons):
            if  speed_button.rect.collidepoint(event.pos) and self.speed_isActive:
              self.selected_speed = speeds[index]
              self.fps = fps[index]
              self.speed_isActive = False
              self.game1 = SnakeGameAI(w=360, h=600, fps=self.fps, surface=self.snake_surface1)
              self.game2 = SnakeGameAI(w=360, h=600, fps=self.fps, surface=self.snake_surface2)

        
        
        mouse_pos = pygame.mouse.get_pos()

        if self.play_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        if self.reset_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND) 
        if self.version_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND) 
        if self.speed_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND) 
        for index, (version_button, v) in enumerate(self.version_buttons):
            if  version_button.rect.collidepoint(mouse_pos) and self.version_isActive:
              pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        for index, (speed_button, v) in enumerate(self.speed_buttons):
            if  speed_button.rect.collidepoint(mouse_pos) and self.speed_isActive:
              pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
              
    def draw(self, screen):
        screen.fill((170,204,153))
   
      
        self.font.render_to(screen,(23,388),
          "VERSION",fgcolor=(34,34,34),bgcolor=None,size=24)
        self.font.render_to(screen,(62,455),
          "v"+str(self.selected_version),fgcolor=(34,34,34),bgcolor=None,size=14)
        self.font.render_to(screen,(35,547),
          "SPEED:",fgcolor=(34,34,34),bgcolor=None,size=24)
        self.font.render_to(screen,(62,614),
          str(self.selected_speed)+"x",fgcolor=(34,34,34),bgcolor=None,size=14)
        
        title = pygame.image.load('src/GameMenu/assets/game/game_title.png').convert_alpha()
        
        # FIX: Update container sizes to match new dimensions
        
      
        screen.blit(title,(17,24))
        screen.blit(self.play_img,(32,203))
        screen.blit(self.reset_img,(32,278))
        screen.blit(self.version_img,(32,437))
        screen.blit(self.speed_img,(32,596))


        self.font.render_to(screen,(229,47),
          "Deep Q-Learning",fgcolor=(34,34,34),bgcolor=None,size=16)
        self.font.render_to(screen,(619,45),
          "Duel Deep Q-Learning",fgcolor=(34,34,34),bgcolor=None,size=16)
    

        # For Versions Dropdown 
        if self.version_isActive:
            for version_button,v in self.version_buttons:
              version_button.draw(screen)
              
        if self.speed_isActive:
            for speed_button,v in self.speed_buttons:
                speed_button.draw(screen)
                
        state1 = self.game1.get_state()
        state2 = self.game2.get_state()
        def get_action(model, state):
            """Convert model prediction into a one-hot action."""
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = model(state_tensor)

            move = torch.argmax(prediction).item()

            action = [0, 0, 0]
            action[move] = 1
            return action

        decision1 = get_action(self.agent1, state1)
        decision2 = get_action(self.agent2, state2)
        # Step the game
        if not(self.game1.game_over) and (self.isPlaying):           
          _, game_over, score = self.game1.play_step(decision1)
          self.score1 = score
             # Step the game
        if not(self.game2.game_over) and (self.isPlaying):           
          _, game_over, score = self.game2.play_step(decision2)
          self.score2 = score

        
        container_x = 209
        container_x_2 = 604
        container_y = 30
        container_width = 370
        container_height = 680
        border_color = (32, 32, 32)
        border_width = 5

        # Top border
        pygame.draw.line(screen, border_color, 
                        (container_x, container_y), 
                        (container_x + container_width, container_y), border_width)

        # Bottom border
        pygame.draw.line(screen, border_color, 
                        (container_x, container_y + container_height), 
                        (container_x + container_width, container_y + container_height), border_width)

        # Left border
        pygame.draw.line(screen, border_color, 
                        (container_x, container_y), 
                        (container_x, container_y + container_height), border_width)

        # Right border
        pygame.draw.line(screen, border_color, 
                        (container_x + container_width, container_y), 
                        (container_x + container_width, container_y + container_height), border_width)
        
        pygame.draw.line(screen, border_color, 
                        (container_x_2, container_y), 
                        (container_x_2 + container_width, container_y), border_width)

        # Bottom border
        pygame.draw.line(screen, border_color, 
                        (container_x_2, container_y + container_height), 
                        (container_x_2 + container_width, container_y + container_height), border_width)

        # Left border
        pygame.draw.line(screen, border_color, 
                        (container_x_2, container_y), 
                        (container_x_2, container_y + container_height), border_width)

        # Right border
        pygame.draw.line(screen, border_color, 
                        (container_x_2 + container_width, container_y), 
                        (container_x_2 + container_width, container_y + container_height), border_width)

        self.font.render_to(screen,(229,83),
          "SCORE:" + str(self.score1),fgcolor=(34,34,34),bgcolor=None,size=16)
        self.font.render_to(screen,(619,83),
          "SCORE:" + str(self.score2),fgcolor=(34,34,34),bgcolor=None,size=16)