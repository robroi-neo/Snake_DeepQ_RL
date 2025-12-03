import pygame
pygame.init()

screen = pygame.display.set_mode((1024, 768))
running = True
while running:
  screen.fill((170,204,153))
  for event in pygame.event.get():
      if event.type == pygame.QUIT:
          running = False
  pygame.display.flip()
pygame.quit()
