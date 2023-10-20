import folium
from folium.plugins import Draw

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

import os

from haversine import haversine, Unit

from shapely.geometry import Point, Polygon

def write_json(lst, file_name):
    with open(file_name, "w") as f:
        json.dump(lst, f)

# Chosen values
height = 100 # m

# Sony RX1R II
focal_length = 35.0 # mm
V = 2
H = 3
aspect_ratio = V/H
sensor_w = 35.9 # mm
sensor_h = 24.0 # mm
num_pixels = 42.4 * pow(10, 6)
pixel_w = num_pixels / aspect_ratio
pixel_h = num_pixels / pixel_w

################################ CALCULATING FOV FROM HIEGHT ################################################################################################
GSD_W = (height * sensor_w) / (focal_length * pixel_w) # m
GSD_H = (height * sensor_h) / (focal_length * pixel_h) # m

FOV_W = GSD_W * pixel_w
FOV_H = GSD_H * pixel_h

print(FOV_W, FOV_H)

path = r"E:\Stephano\Documents\Stephano-MIng"

src_file = r"E:\Stephano\Downloads\search_area.geojson"
dst_file = r"E:\Stephano\Documents\Stephano-MIng\search_area.geojson"
os.rename(src_file, dst_file)

# Load the .geojson file
with open("search_area.geojson", "r") as f:
    data = json.load(f)

# Extract the title and coordinates from the file
title = data["features"][0]["properties"]["Title"]
ls_coordinates = data["features"][0]["geometry"]["coordinates"]

# Convert shape from (1,N,2) to (N,2)
coordinates = np.array(ls_coordinates)
coordinates = np.squeeze(coordinates)

# Swap latitude and longitude positions
coordinates[:, [1,0]] = coordinates[:,[0,1]]

ls_coordinates = coordinates.tolist()

print("Title: ", title)
print("Coordinates: ", ls_coordinates)

# Create folium map
m = folium.Map(location=ls_coordinates[0], zoom_start=13)

# Add polygon to map
folium.Polygon(ls_coordinates).add_to(m)

# Latitude is the Y axis, longitude is the X axis
# Get maximum and minimum latitude and longitude coordinates of polygon
max_lat_index = np.argmax(coordinates[:,0])
min_lat_index = np.argmin(coordinates[:,0])
max_long_index = np.argmax(coordinates[:,1])
min_long_index = np.argmin(coordinates[:,1])

# Get distance between max and min lat and long in meters
lat_dist = haversine(coordinates[max_lat_index], coordinates[min_lat_index], unit=Unit.METERS)
long_dist = haversine(coordinates[max_long_index], coordinates[min_long_index], unit=Unit.METERS)

# Get size of grid based on FOV of drones
grid_width = round(lat_dist / FOV_W)
grid_height = round(long_dist / FOV_H)
print(grid_width, grid_height)

grid = np.zeros((grid_height, grid_width))

# Get distance of lat and long in meters
diff_lat = coordinates[max_lat_index][0] - coordinates[min_lat_index][0]
diff_long = coordinates[max_long_index][1] - coordinates[min_long_index][1]

# Get cell size
cell_h = diff_lat / grid_height
cell_w = diff_long / grid_width
print(cell_w, cell_h)

# Starting coordinates (top left) of grid
coordinate0 = [coordinates[max_lat_index][0], coordinates[min_long_index][1]]
coordinate1 = [(coordinates[max_lat_index][0]-cell_h), (coordinates[min_long_index][1]+cell_w)]

# create a polygon object with the coordinates of the polygon
polygon = Polygon(ls_coordinates)

# Draw grid on map
for i in range(0, grid_height):
    for j in range(0, grid_width):
        # create a point object with the coordinates of the cell
        point0 = Point([coordinate0[0], coordinate0[1]])
        point1 = Point([coordinate0[0], coordinate0[1]+cell_w])
        point2 = Point([coordinate0[0]-cell_h, coordinate0[1]+cell_w])
        point3 = Point([coordinate0[0]-cell_h, coordinate0[1]])
        
        # check if the rectangle is inside the polygon and draw it
        if polygon.contains(point0) or polygon.contains(point1) or polygon.contains(point2) or polygon.contains(point3):
            folium.Rectangle([coordinate0, coordinate1], color='black', weight=0.5).add_to(m)
            grid[i,j] = 0
        else:
            folium.Rectangle([coordinate0, coordinate1], color='black', weight=0.5, fill=True).add_to(m)
            grid[i,j] = 1

        coordinate0[1] += cell_w
        coordinate1[1] += cell_w
    coordinate0[1] = coordinates[min_long_index][1]
    coordinate1[1] = coordinates[min_long_index][1]+cell_w
    coordinate0[0] -= cell_h
    coordinate1[0] -= cell_h

file_name = title + '.html'

m.save(file_name)

file_name = "grid.json"
file_name = os.path.join(path, file_name)
write_json([grid.tolist()], file_name)

print("Grid shape: ",grid.shape, "\n", grid)