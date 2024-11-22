import os
import numpy as np
import matplotlib.pyplot as plt
import pygame
import math
import sys
from typing import Literal
from tqdm import tqdm


# Constants for julia plot
MAXITER = 100
NUM_CDS = 200  # Number of coordinates
CD_MIN = -2.0
CD_MAX = 2.0

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


def julia(j_p: complex, c: complex) -> int:
    # Global variable used: MAXITER
    # R is +ve root of r^2 - r - |j_p| = 0
    R = (1 + (1 + 4 * abs(j_p))**0.5 ) / 2.0
    # n is used for the number of steps/iterations
    n = 0
    # z starts life as z_0 = c
    z = c
    # Compute z = z_n at each iteration and continue for as long
    # as the sequence (z_n) does not start to diverge and n < MAXITER
    while abs(z) <= R and n < MAXITER:
        z = z**2 + j_p
        n += 1
    # Then return the first n such that we escape the threshold R (or else n = MAXITER)
    return n


def make_julia_data(
    j_p: complex,
    cd_min: float,
    cd_max: float,
    num_cds: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates grid data for a Julia set.

    Parameters:
    - j_p (complex): The Julia set parameter.
    - cd_min (float): The minimum value for both x and y coordinates.
    - cd_max (float): The maximum value for both x and y coordinates.
    - num_cds (int): The number of equally spaced coordinates in each dimension.

    Returns:
    - tuple: A tuple containing three NumPy arrays:
        - x_matrix (ndarray): A 2D array of x coordinates.
        - y_matrix (ndarray): A 2D array of y coordinates.
        - n_matrix (ndarray): A 2D array with the results of the julia function.
    """

    # Generate equally spaced coordinates for x and y axes
    x = np.linspace(cd_min, cd_max, num_cds)
    y = np.linspace(cd_min, cd_max, num_cds)

    # Create a meshgrid from the x and y coordinates
    x_matrix, y_matrix = np.meshgrid(x, y)

    # Combine x and y matrices to form complex numbers for each grid point
    c_matrix = x_matrix + 1j * y_matrix

    # Initialize the n_matrix to store results from the julia function
    # The data type can be adjusted based on what the julia function returns
    n_matrix = np.zeros((num_cds, num_cds), dtype=int)

    # Iterate over each point in the grid and apply the julia function
    for i in range(num_cds):
        for j in range(num_cds):
            c = c_matrix[i, j]
            n_matrix[i, j] = julia(j_p, c)

    return x_matrix, y_matrix, n_matrix


def generate_julia_animation_frames(
    num_frames: int, 
    output_folder: str = 'julia_frames', 
    colormap: Literal['plasma', 'magma', 'viridis', 'inferno'] = 'inferno'
):
    os.makedirs(output_folder, exist_ok=True)
    angles = np.linspace(0, 2 * math.pi, num_frames, endpoint=False)
    for idx in tqdm(range(len(angles))):
        j_p = 0.7885 * np.exp(1j * angles[idx])
        x_matrix, y_matrix, n_matrix = make_julia_data(j_p, CD_MIN, CD_MAX, NUM_CDS)
        plt.figure(figsize=(6, 6))
        plt.pcolor(x_matrix, y_matrix, n_matrix, cmap=colormap, vmin=0, vmax=MAXITER, shading='auto')
        plt.axis('off')
        title_str = f"Julia Set\nj_p = {j_p.real:.4f} + {j_p.imag:.4f}i"
        plt.title(title_str, fontsize=12)
        filename = os.path.join(output_folder, f"frame_{idx:03d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    print(f"All {num_frames} frames have been generated in the '{output_folder}' folder.")


def load_images(image_folder: str, num_frames: int) -> list[pygame.Surface]:
    images = []
    for idx in range(num_frames):
        filename = os.path.join(image_folder, f"frame_{idx:03d}.png")
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist.")
            continue
        image = pygame.image.load(filename)
        images.append(image)
    return images


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


def animate_julia_sets(
    image_folder: str, 
    num_frames: int, 
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
    
    # Load images
    print("Loading images...")
    images = load_images(image_folder, num_frames)
    if not images:
        print("No images loaded. Exiting program.")
        pygame.quit()
        sys.exit()
    print(f"Loaded {len(images)} image frames.")
    
    # Get image dimensions from the first image
    image_width, image_height = images[0].get_size()
    
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
            screen.fill((255, 255, 255))

            # Display the current frame image
            screen.blit(images[frame_idx], (0, 105))  # Reserve top space for the slider
            
            # Move to the next frame
            frame_idx = (frame_idx + 1) % len(images)
        
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
        
        # Render and display the minimum and maximum FPS labels
        # min_fps_text = font.render(str(MIN_FPS), True, (255, 255, 255))
        # max_fps_text = font.render(str(MAX_FPS), True, (255, 255, 255))
        # screen.blit(min_fps_text, (slider_x - 10, slider_y + SLIDER_HEIGHT + 5))
        # screen.blit(max_fps_text, (slider_end_x - max_fps_text.get_width() + 10, slider_y + SLIDER_HEIGHT + 5))
        
        # Update the display
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(fps)
    
    # Quit Pygame
    pygame.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Julia Set Animation Generator and Viewer")
    parser.add_argument(
        '--generate', 
        action='store_true', 
        help="Generate Julia set frames."
    )
    parser.add_argument(
        '--num_frames', 
        type=int, 
        default=200, 
        help="Number of frames to generate or animate."
    )
    parser.add_argument(
        '--output_folder', 
        type=str, 
        default='julia_frames', 
        help="Folder to save or load frames."
    )
    parser.add_argument(
        '--colormap', 
        type=str, 
        choices=['plasma', 'magma', 'viridis', 'inferno'],
        default='inferno', 
        help="Colormap chosen from 'plasma', 'magma', 'viridis' and 'inferno'."
    )

    args = parser.parse_args()

    generate = args.generate
    if not os.path.exists(args.output_folder):
        generate = True
    
    if generate:
        print("Starting frame generation...")
        generate_julia_animation_frames(
            num_frames=args.num_frames, 
            output_folder=args.output_folder, 
            colormap=args.colormap
        )
    
    print("Starting animation...")
    animate_julia_sets(
        image_folder=args.output_folder, 
        num_frames=args.num_frames, 
        initial_speed=60
    )


if __name__ == '__main__':
    main()