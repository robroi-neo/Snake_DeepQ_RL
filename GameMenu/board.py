import pygame
import pygame.freetype
from screen_base import ScreenBase
pygame.init()



class MenuScreen(ScreenBase):
    def __init__(self, game):
        super().__init__(game)
        self.font = pygame.freetype.Font('GameMenu/assets/menu/PressStart2p-Regular.ttf', 50)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.game.change_screen("game") 
    def draw(self, screen):
      screen.fill((170,204,153))
      title = pygame.image.load('GameMenu/assets/menu/menu_title.png').convert_alpha()
      play_button = pygame.image.load('GameMenu/assets/menu/play_button.png').convert_alpha()
      quit_button = pygame.image.load('GameMenu/assets/menu/quit_button.png').convert_alpha()
      
      self.font.render_to(screen,(442,692),
          "DISCLAIMER",fgcolor=(34,34,34),bgcolor=None,size=14)
      
      # Claims Part 1
      self.font.render_to(screen,(102,720),
          "THIS PROJECT DEMONSTRATES HOW REINFORCEMENT" \
          " LEARNING ALLOWS AN AGENT TO LEARN AND ",
          fgcolor=(34,34,34),bgcolor=None,size=10)
      # Claims Part 2
      self.font.render_to(screen,(224,734),
          "OPTIMIZE ITS STRATEGY IN THE CLASSIC SNAKE ENVIRONMENT",
          fgcolor=(34,34,34),bgcolor=None,size=10)
      screen.blit(title,(116,120))
      screen.blit(play_button,(412,452))
      screen.blit(quit_button,(412,552))
  