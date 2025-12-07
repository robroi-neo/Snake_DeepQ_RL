import pygame
import pygame.freetype

class RectWithText:
    def __init__(self, x, y, w, h, text, font_size=24,
                 text_color=(0,0,0), fill_color=(200,200,200),
                 border_color=(0,0,0), border_width=3, center=True):

        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.fill_color = fill_color
        self.border_color = border_color
        self.border_width = border_width
        self.center = center

        self.font = pygame.freetype.Font("src/GameMenu/assets/menu/PressStart2p-Regular.ttf", font_size)
        self.text_color = text_color

    def draw(self, screen):
        # Draw the box
        pygame.draw.rect(screen, self.fill_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, self.border_width)

        # Render the text surface + get bounding box
        surf, rect = self.font.render(self.text, self.text_color)

        if self.center:
            # Center text inside the rect
            rect.center = self.rect.center
        else:
            # Draw text at left side with padding
            rect.topright = (self.rect.right - 18, self.rect.y + 18)

        # Draw text
        screen.blit(surf, rect)
