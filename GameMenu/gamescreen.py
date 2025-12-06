import pygame
import pygame.freetype
from screen_base import ScreenBase

from rect import RectWithText
pygame.init()

clock = pygame.time.Clock()

versions = [1,20,40,80]

class GameScreen(ScreenBase):
    def __init__(self, game):
        super().__init__(game)
        self.font = pygame.freetype.Font('GameMenu/assets/menu/PressStart2p-Regular.ttf', 50)
        self.play_img = pygame.image.load('GameMenu/assets/game/play.png').convert_alpha()
        self.reset_img = pygame.image.load('GameMenu/assets/game/reset.png').convert_alpha()
        self.version_img = pygame.image.load('GameMenu/assets/game/dropdown.png').convert_alpha()
        self.speed_img = pygame.image.load('GameMenu/assets/game/dropdown.png').convert_alpha()
        self.play_rect = self.play_img.get_rect(topleft=(32, 203))
        self.reset_rect = self.reset_img.get_rect(topleft=(32, 278))
        self.version_rect = self.version_img.get_rect(topleft=(32,437))
        self.version_buttons = []
        self.selected_version = versions[0]
        self.version_isActive = False

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

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.game.change_screen("game") 

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # left click
          if self.play_rect.collidepoint(event.pos):
                self.game.change_screen("game")
          if self.reset_rect.collidepoint(event.pos):
                pygame.quit()
                exit()
          if self.version_rect.collidepoint(event.pos):
                self.version_isActive = not self.version_isActive
          for index, (version_button, v) in enumerate(self.version_buttons):
            if  version_button.rect.collidepoint(event.pos) and self.version_isActive:
              self.selected_version = versions[index]
              self.version_isActive = False
        
        mouse_pos = pygame.mouse.get_pos()

        if self.play_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        if self.reset_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND) 
        if self.version_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND) 
        for index, (version_button, v) in enumerate(self.version_buttons):
            if  version_button.rect.collidepoint(mouse_pos) and self.version_isActive:
              pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
    def draw(self, screen):
        screen.fill((170,204,153))
   
      
        self.font.render_to(screen,(23,388),
          "VERSION",fgcolor=(34,34,34),bgcolor=None,size=24)
        self.font.render_to(screen,(68,455),
          "v"+str(self.selected_version),fgcolor=(34,34,34),bgcolor=None,size=14)
        self.font.render_to(screen,(35,547),
          "SPEED:",fgcolor=(34,34,34),bgcolor=None,size=24)
        self.font.render_to(screen,(68,614),
          "2x",fgcolor=(34,34,34),bgcolor=None,size=14)
        
        title = pygame.image.load('GameMenu/assets/game/game_title.png').convert_alpha()
        screen.blit(title,(17,24))
        screen.blit(self.play_img,(32,203))
        screen.blit(self.reset_img,(32,278))
        screen.blit(self.version_img,(32,437))
        screen.blit(self.speed_img,(32,596))
       
      
        # For Versions Dropdown 
        if self.version_isActive:
            for version_button, v in self.version_buttons:
              
              version_button.draw(screen)
          
    
      
        
    