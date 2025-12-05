import pygame
import pygame.freetype

pygame.init()
screen = pygame.display.set_mode((1020, 780))
pygame.display.set_caption("Input Box with Blinking Cursor")
clock = pygame.time.Clock()

class InputBox:
    def __init__(self, x, y, w, h, font_size=32, text_color=(255,255,255), box_color=(50,50,50), border_color=(255,255,255)):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = box_color
        self.border_color = border_color
        self.text_color = text_color
        self.font = pygame.freetype.Font(None, font_size)
        self.text = ""
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_interval = 500  # milliseconds

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state if clicked inside box
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False

        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                print("Entered:", self.text)
                self.text = ""
            else:
                self.text += event.unicode
      
    def update(self, dt):
        if self.active:
            self.cursor_timer += dt
            if self.cursor_timer >= self.cursor_interval:
                self.cursor_timer = 0
                self.cursor_visible = not self.cursor_visible
        else:
            self.cursor_visible = False

    def draw(self, screen):
        # Draw input box
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)

        # Draw text

        # Draw blinking cursor
        if self.active and self.cursor_visible:
            text_width, _ = self.font.get_rect(self.text)[2:4]
            cursor_x = self.rect.x + 5 + text_width
            cursor_y = self.rect.y + 5
            cursor_height = self.font.size
            pygame.draw.line(screen, self.text_color, (cursor_x, cursor_y), (cursor_x, cursor_y + cursor_height), 2)
        self.font.render_to(screen, (self.rect.x + 5, self.rect.y + 5), self.text, self.text_color)
