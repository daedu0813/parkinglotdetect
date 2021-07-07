from sys import byteorder
from coordinates_generator import CoordinatesGenerator
from colors import *
import logging

def main():
    logging.basicConfig(level=logging.INFO)

    image_file = 'Setting Parking Space'
    data_file = 'data/data.yml'

    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = CoordinatesGenerator(image_file, points, COLOR_RED)
            generator.generate()

if __name__ == '__main__':
    main()