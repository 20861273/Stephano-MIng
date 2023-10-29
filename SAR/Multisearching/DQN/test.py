from enclosed_space_checker import Enclosed_space_check
import numpy as np
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
import time

# # Define the number of rows and columns in the grid
# num_rows, num_cols = 3,3

# # Generate a grid of cell coordinates
# x, y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
# cell_coordinates = np.stack((x.ravel(), y.ravel()), axis=1)

# # Calculate Manhattan distances between all cells using vectorized operations
# distances = np.sum(np.abs(cell_coordinates[:, None, :] - cell_coordinates[None, :, :]), axis=2)

# print(distances)

# # Define the grid size and create a grid of cell coordinates
# start = time.time()
# num_rows, num_cols = 100, 100
# x, y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
# cell_coordinates = np.stack((x.ravel(), y.ravel()), axis=1)

# # Calculate Manhattan distances without obstacles
# distances_without_obstacles = np.abs(cell_coordinates[:, None, :] - cell_coordinates[None, :, :]).sum(axis=2)

# # Define obstacle coordinates (row, column)
# obstacle_coordinates = [(0, 1)]

# # Create a grid of obstacle coordinates
# obstacles = np.zeros((num_rows, num_cols), dtype=bool)
# for obstacle in obstacle_coordinates:
#     obstacles[obstacle[0], obstacle[1]] = True

# # Update distances considering obstacles
# distances_with_obstacles = distances_without_obstacles.copy()
# for obstacle in obstacle_coordinates:
#     obstacle_index = obstacle[0] * num_cols + obstacle[1]
#     distances_with_obstacles[:, obstacle_index] = distances_with_obstacles[obstacle_index, :] = -1
#     for i in range(num_rows * num_cols):
#         row, col = i // num_cols, i % num_cols
#         if obstacles[row, col]:
#             distances_with_obstacles[:, i] = distances_with_obstacles[i, :] = -1

# def calculate_distances_with_obstacles(num_rows, num_cols, obstacles):
#     num_cells = num_rows * num_cols
#     distances = np.full((num_cells, num_cells), -1)  # Initialize distances to -1 (unreachable)

#     # Define movements (up, down, left, right)
#     moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

#     for start_cell in range(num_cells):
#         start_row, start_col = start_cell // num_cols, start_cell % num_cols
#         if (start_row, start_col) not in obstacles:
#             queue = deque([(start_cell, 0)])  # Initialize queue with the starting cell and distance 0
#             visited = np.zeros(num_cells, dtype=bool)  # Mark all cells as not visited
#             visited[start_cell] = True  # Mark the starting cell as visited

#             while queue:
#                 current_cell, distance = queue.popleft()
#                 distances[start_cell, current_cell] = distance

#                 # Explore neighbors
#                 row, col = current_cell // num_cols, current_cell % num_cols
#                 for move_row, move_col in moves:
#                     new_row, new_col = row + move_row, col + move_col
#                     if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
#                         neighbor_cell = new_row * num_cols + new_col
#                         if (new_row, new_col) not in obstacles and not visited[neighbor_cell]:
#                             queue.append((neighbor_cell, distance + 1))
#                             visited[neighbor_cell] = True

#     return distances

# def draw_distance_grid(distances, num_rows, num_cols):
#     distances_matrix = distances.reshape(num_rows, num_cols)
    
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(distances_matrix, cmap='viridis')

#     # Set labels and ticks
#     ax.set_xticks(np.arange(num_cols))
#     ax.set_yticks(np.arange(num_rows))
#     ax.set_xticklabels(np.arange(num_cols))
#     ax.set_yticklabels(np.arange(num_rows))

#     # Show the distances as text annotations within each cell
#     for i in range(num_rows):
#         for j in range(num_cols):
#             text = ax.text(j, i, str(distances_matrix[i, j]),
#                            ha="center", va="center", color="w")

#     # Display a colorbar to represent distance values
#     cbar = ax.figure.colorbar(im, ax=ax, cmap='viridis')
#     cbar.set_label('Distances')

#     ax.set_title('Distances from Cell (0, 0)')
#     plt.xlabel('Column')
#     plt.ylabel('Row')
#     plt.show()

# # Example usage
# num_rows, num_cols = 3, 3
# obstacle_coordinates = [(0, 1), (1,1)]  # Obstacle coordinates as (row, col)

# start = time.time()
# distances_with_obstacles = calculate_distances_with_obstacles(num_rows, num_cols, obstacle_coordinates)
# end = time.time()
# print("Distances with Obstacles:")
# print(distances_with_obstacles)
# print(end-start)

# draw_distance_grid(distances_with_obstacles[0], num_rows, num_cols)
# breakpoint

list = [7, 5, 6]

if all([l==None for l in list]):
    print("hoi")