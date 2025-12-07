import pygame
import pygame.freetype
from screen_base import ScreenBase

from rect import RectWithText
pygame.init()

clock = pygame.time.Clock()

versions = [1,20,40,80]
speeds = [1,2,4,8,16]
class GameScreen(ScreenBase):
    def __init__(self, game):
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
          if self.version_rect.collidepoint(event.pos) and not self.speed_isActive:
                self.version_isActive = not self.version_isActive
          if self.speed_rect.collidepoint(event.pos) and not self.version_isActive:
                self.speed_isActive = not self.speed_isActive
          for index, (version_button, v) in enumerate(self.version_buttons):
            if  version_button.rect.collidepoint(event.pos) and self.version_isActive:
              self.selected_version = versions[index]
              self.version_isActive = False
          for index, (speed_button, v) in enumerate(self.speed_buttons):
            if  speed_button.rect.collidepoint(event.pos) and self.speed_isActive:
              self.selected_speed = speeds[index]
              self.speed_isActive = False
        
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
        self.font.render_to(screen,(68,455),
          "v"+str(self.selected_version),fgcolor=(34,34,34),bgcolor=None,size=14)
        self.font.render_to(screen,(35,547),
          "SPEED:",fgcolor=(34,34,34),bgcolor=None,size=24)
        self.font.render_to(screen,(68,614),
          str(self.selected_speed)+"x",fgcolor=(34,34,34),bgcolor=None,size=14)
        
        title = pygame.image.load('src/GameMenu/assets/game/game_title.png').convert_alpha()
        leftContainer = RectWithText(
                    214,
                    30,
                    390,
                    706,
                    "",
                    14,
                    fill_color=(170,204,153),
                    text_color=(32,32,32),
                    border_width=5,
                    border_color=(32,32,32),
                    center=False
        )
        rightContainer = RectWithText(
                    604,
                    30,
                    390,
                    706,
                    "",
                    14,
                    fill_color=(170,204,153),
                    text_color=(32,32,32),
                    border_width=5,
                    border_color=(32,32,32),
                    center=False
        )
        screen.blit(title,(17,24))
        screen.blit(self.play_img,(32,203))
        screen.blit(self.reset_img,(32,278))
        screen.blit(self.version_img,(32,437))
        screen.blit(self.speed_img,(32,596))
        leftContainer.draw(screen)
        rightContainer.draw(screen)

        self.font.render_to(screen,(229,47),
          "Deep Q-Learning",fgcolor=(34,34,34),bgcolor=None,size=16)
        self.font.render_to(screen,(619,45),
          "Brand X",fgcolor=(34,34,34),bgcolor=None,size=16)
        self.font.render_to(screen,(229,83),
          "SCORE:",fgcolor=(34,34,34),bgcolor=None,size=16)
        self.font.render_to(screen,(619,83),
          "SCORE:",fgcolor=(34,34,34),bgcolor=None,size=16)


        # For Versions Dropdown 
        if self.version_isActive:
            for version_button,v in self.version_buttons:
              
              version_button.draw(screen)
              
        if self.speed_isActive:
            for speed_button,v in self.speed_buttons:
                speed_button.draw(screen)

        # PLACE YOUR CODE HERE TO MAKE SURE THAT THE SNAKE IS ON TOP OF THE
        # CONTAINER

        
          
    
      
        
    