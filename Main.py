import pygame
import numpy as np

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 600  # Adjust as needed
screen = pygame.display.set_mode((screen_width, screen_height))

def value_to_color(value):
    if value <= -1:
        return (0, 0, 255)  # Blue
    elif value >= 1:
        return (255, 0, 0)  # Red
    else:
        # Intermediate values shown as a mix of blue and red
        return (int(255 * (value + 1) / 2), 0, int(255 * (1 - value) / 2))

def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    current_step = None
    current_layer = None

    for line in lines:
        line = line.strip()
        if line.startswith('Epoch:'):
            if current_step is not None:
                data.append(current_step)
            current_step = {'input_layer': [], 'middle_layer': [], 'output_layer': []}
        elif line.startswith('input_layer') or line.startswith('middle_layer') or line.startswith('output_layer'):
            current_layer = line.split(':')[0]
        elif line.startswith('['):
            values = eval(line)
            current_step[current_layer].append(values)

    if current_step is not None:
        data.append(current_step)

    return data

def draw_layers(data, step, screen):
    screen.fill((0, 0, 0))  # Clear screen

    # Initialize font
    font = pygame.font.SysFont(None, 36)
    text = font.render(f'Time Step: {step}', True, (255, 255, 255))
    screen.blit(text, (10, 10))  # Position the text at the top-left corner

    layers = ['input_layer', 'middle_layer', 'output_layer']
    layer_sizes = [(3, 3), (6, 6), (2, 1)]  # Assuming fixed sizes, adjust as necessary
    margin = 50
    total_width = screen_width - margin * 2
    width_per_layer = total_width // len(layers)

    for i, layer in enumerate(layers):
        layer_data = data[step][layer]
        rows, cols = layer_sizes[i]
        cell_size = min(width_per_layer // cols, (screen_height - margin * 2) // rows)
        start_x = margin + i * width_per_layer + (width_per_layer - cols * cell_size) // 2
        start_y = margin + (screen_height - rows * cell_size) // 2

        for row in range(rows):
            for col in range(cols):
                value = layer_data[row][col] if row < len(layer_data) and col < len(layer_data[row]) else 0
                color = value_to_color(value)
                pygame.draw.rect(screen, color, (start_x + col * cell_size, start_y + row * cell_size, cell_size, cell_size))

def run_simulation(data):
    clock = pygame.time.Clock()
    running = True
    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_layers(data, step, screen)
        pygame.display.flip()
        step = (step + 1) % len(data)
        clock.tick(10)  # Update up to 10 times per second

    pygame.quit()

data = load_data('history.txt')
run_simulation(data)
