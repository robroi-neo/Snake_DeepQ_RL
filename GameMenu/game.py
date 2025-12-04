import pygame
from screen_base import ScreenBase
from  board import MenuScreen 
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 768))
        pygame.display.set_caption("Screen System")

        # all screens stored here
        self.screens = {
            "menu": MenuScreen(self),
            # "game": GameScreen(self)
        }

        self.current_screen = self.screens["menu"]

    def change_screen(self, name):
        self.current_screen = self.screens[name]

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                self.current_screen.handle_event(event)

            self.current_screen.update()
            self.current_screen.draw(self.screen)

            pygame.display.flip()

        pygame.quit()


game = Game()
game.run()