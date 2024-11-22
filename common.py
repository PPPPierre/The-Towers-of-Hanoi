import pygame


# Constants for slider dimensions and positions
MIN_FPS = 1
MAX_FPS = 120
SLIDER_WIDTH = 250
SLIDER_HEIGHT = 10
SLIDER_POS = (50, 50)  # Position of the slider track (top-left corner)
KNOB_RADIUS = 10
BACKGROUND_COLOR = (30, 30, 30)  # Background color for the slider area

# Button properties
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 40
BUTTON_POS = (0, 25)  # Will be updated based on image_width
BUTTON_COLOR = (70, 130, 180)  # Steel Blue
BUTTON_HOVER_COLOR = (100, 149, 237)  # Cornflower Blue
BUTTON_TEXT_COLOR = (255, 255, 255)  # White


class Drawer:
    def __init__(self) -> None:
        pass

    def init_drawer(self, *args, **kwargs):
        raise NotImplementedError()

    def draw(self, screen: pygame.Surface):
        raise NotImplementedError()

    def get_size(self) -> tuple[int, int]:
        raise NotImplementedError()


def render_button_text(paused, font):
    """
    Renders the button text based on the paused state.
    
    Parameters:
    - paused (bool): Whether the animation is paused.
    - font (pygame.font.Font): Font object for rendering text.
    
    Returns:
    - pygame.Surface: Rendered text surface.
    """
    text = "Continue" if paused else "Pause"
    return font.render(text, True, BUTTON_TEXT_COLOR)


def animate(
    drawer, 
    initial_speed: float = 30.0
):
    """
    Displays an animation of Julia sets using Pygame with real-time speed control and pause/play functionality.

    Parameters:
    - image_folder (str): Path to the folder containing image frames.
    - num_frames (int): Number of frames in the animation.
    - initial_speed (float): Initial frames per second (FPS).

    Returns:
    - None
    """
    # Initialize Pygame
    pygame.init()
    
    # Get image dimensions from the first image
    image_width, image_height = drawer.get_size()
    
    # Set up the display window, adding extra height for the slider
    window_height = image_height + 100  # Extra space for the slider

    screen = pygame.display.set_mode((image_width, window_height))
    pygame.display.set_caption("Julia Set Animation with Speed Slider")
    
    # Create a clock object to manage the frame rate
    clock = pygame.time.Clock()
    fps = initial_speed  # Current frame rate
    
    # Animation state variables
    running = True
    paused = False
    frame_idx = 0
    
    # Initialize font for text rendering
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)
    
    # Define slider properties
    slider_x, slider_y = SLIDER_POS
    slider_end_x = slider_x + SLIDER_WIDTH
    knob_x = slider_x + (fps - MIN_FPS) / (MAX_FPS - MIN_FPS) * SLIDER_WIDTH
    knob_y = slider_y + SLIDER_HEIGHT // 2
    dragging = False  # Indicates if the slider knob is being dragged
    
    # Update button position based on image_width
    BUTTON_POS = (slider_end_x + 50, slider_y + SLIDER_HEIGHT // 2 - BUTTON_HEIGHT // 2)  # Positioned top-right with 50px padding
    button_rect = pygame.Rect(BUTTON_POS[0], BUTTON_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)
    button_hover = False  # Indicates if the mouse is hovering over the button
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                # Check if the slider knob is clicked
                distance = ((mouse_x - knob_x)**2 + (mouse_y - knob_y)**2)**0.5
                if distance <= KNOB_RADIUS:
                    dragging = True
                # Check if the button is clicked
                if button_rect.collidepoint(event.pos):
                    paused = not paused  # Toggle pause/play
                    if paused:
                        print("Animation paused.")
                    else:
                        print("Animation playing.")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_x, _ = event.pos
                    # Constrain the knob within the slider track
                    knob_x = max(slider_x, min(mouse_x, slider_end_x))
                    # Update FPS based on knob position
                    fps = MIN_FPS + (knob_x - slider_x) / SLIDER_WIDTH * (MAX_FPS - MIN_FPS)
                    fps = round(fps)
                # Update button hover state
                if button_rect.collidepoint(event.pos):
                    button_hover = True
                else:
                    button_hover = False
        
        if not paused:
            drawer.draw(screen)
        
        # Draw the slider background
        pygame.draw.rect(screen, BACKGROUND_COLOR, (0, 0, image_width, 100))
        
        # Draw the slider track with rounded corners
        pygame.draw.rect(screen, (200, 200, 200), (slider_x, slider_y, SLIDER_WIDTH, SLIDER_HEIGHT), border_radius=5)
        
        # Draw the slider knob with a white border for better visibility
        pygame.draw.circle(screen, (100, 100, 255), (int(knob_x), int(knob_y)), KNOB_RADIUS)
        pygame.draw.circle(screen, (255, 255, 255), (int(knob_x), int(knob_y)), KNOB_RADIUS, 2)  # Knob border
        
        # Draw the pause/continue button
        if button_rect.collidepoint(mouse_pos):
            current_button_color = BUTTON_HOVER_COLOR
        else:
            current_button_color = BUTTON_COLOR
        pygame.draw.rect(screen, current_button_color, button_rect, border_radius=5)
        
        # Render and display the button text
        button_text_surface = render_button_text(paused, font)
        button_text_rect = button_text_surface.get_rect(center=button_rect.center)
        screen.blit(button_text_surface, button_text_rect)
        
        # Render and display the FPS text
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (slider_x, slider_y - 30))
        
        # Update the display
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(fps)
    
    # Quit Pygame
    pygame.quit()