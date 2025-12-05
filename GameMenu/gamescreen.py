import pygame
import pygame.freetype
from screen_base import ScreenBase
from inputbox import InputBox

pygame.init()

clock = pygame.time.Clock()

class GameScreen(ScreenBase):
    def __init__(self, game):
        super().__init__(game)
        self.font = pygame.freetype.Font('GameMenu/assets/menu/PressStart2p-Regular.ttf', 50)
        self.play_img = pygame.image.load('GameMenu/assets/menu/play_button.png').convert_alpha()
        self.quit_img = pygame.image.load('GameMenu/assets/menu/quit_button.png').convert_alpha()
        self.play_rect = self.play_img.get_rect(topleft=(412, 452))
        self.quit_rect = self.quit_img.get_rect(topleft=(412, 552))
        self.input_box = InputBox(32, 437, 150, 50,
        text_color=(34,34,34),box_color=(170,204,153),border_color=(34,34,34))
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
          self.input_box.handle_event(event)
    def draw(self, screen):
        dt = clock.tick(60)  # milliseconds since last frame
        screen.fill((170,204,153))
      
        self.input_box.update(dt)
        self.input_box.draw(screen)
        mouse_pos = pygame.mouse.get_pos()
        title = pygame.image.load('GameMenu/assets/game/game_title.png').convert_alpha()
        screen.blit(title,(17,24))
      
        
    