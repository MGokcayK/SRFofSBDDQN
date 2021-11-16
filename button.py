import pygame

class Button(object):
    def __init__(self,
                    text, font, 
                    text_color, rect, idle_color,
                    hover_color, callback_function):
        self.text = text
        self.label = font.render(text, True, text_color)
        self.rect = pygame.Rect(rect)
        #we'll use this rect to center the text on the button
        self.label_rect = self.label.get_rect(center=self.rect.center)
        self.idle_color = idle_color
        self.hover_color = hover_color
        self.hovered = False
        self.callback = callback_function
        
        
    def update(self, mouse_pos):
        self.hovered = False
        if self.rect.collidepoint(mouse_pos):
            self.hovered = True
        
    def get_event(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            if self.hovered:
                self.callback()
                
    def draw(self, surface):
        #determine draw color based on whether mouse is over button
        color = self.hover_color if self.hovered else self.idle_color
        pygame.draw.rect(surface, color, self.rect)
        surface.blit(self.label, self.label_rect)