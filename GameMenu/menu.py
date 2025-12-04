import pygame
import pygame.freetype
from screen_base import ScreenBase
pygame.init()



class MenuScreen(ScreenBase):
    def __init__(self, game):
        super().__init__(game)
        self.font = pygame.freetype.Font('GameMenu/assets/menu/PressStart2p-Regular.ttf', 50)
        self.play_img = pygame.image.load('GameMenu/assets/menu/play_button.png').convert_alpha()
        self.quit_img = pygame.image.load('GameMenu/assets/menu/quit_button.png').convert_alpha()
        self.play_rect = self.play_img.get_rect(topleft=(412, 452))
        self.quit_rect = self.quit_img.get_rect(topleft=(412, 552))
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.game.change_screen("game") 

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # left click
          if self.play_rect.collidepoint(event.pos):
                self.game.change_screen("game")
          if self.quit_rect.collidepoint(event.pos):
                pygame.quit()
                exit()
               
    def draw(self, screen):
        screen.fill((170,204,153))
        mouse_pos = pygame.mouse.get_pos()
      
        title = pygame.image.load('GameMenu/assets/menu/menu_title.png').convert_alpha()

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
        screen.blit(self.play_img,(412,452))
        screen.blit(self.quit_img,(412,552))
        if self.play_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        if self.quit_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)   
    