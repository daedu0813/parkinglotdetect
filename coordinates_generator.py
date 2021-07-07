import cv2 as open_cv
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
from stream import *

from colors import COLOR_WHITE
from drawing_utils import draw_contours


class CoordinatesGenerator:
    KEY_SAVE = ord("s")

    def __init__(self, image, output, color):
        self.output = output
        self.caption = image
        self.color = color

        #url = IMAGE
        #res = request.urlopen(url).read()
        #img = Image.open(BytesIO(res)).convert('RGB')
        #open_cv_image = np.array(img) 
        # Convert RGB to BGR 
        #open_cv_image = open_cv_image[:, :, ::-1].copy() 
        #self.image = open_cv_image.copy()

        self.image = open_cv.imread(IMAGE).copy()

        self.click_count = 0
        self.ids = 0
        self.coordinates = []

        open_cv.namedWindow(self.caption, open_cv.WINDOW_GUI_EXPANDED)
        open_cv.setMouseCallback(self.caption, self.__mouse_callback)

    def generate(self):
        while True:
            open_cv.imshow(self.caption, self.image)
            print("클릭으로 주차 공간을 설정해주세요.")
            print("주차 공간 설정을 저장 하려면 [S]키를 눌러주세요.")
            key = open_cv.waitKey(0)
            
            if key == CoordinatesGenerator.KEY_SAVE:
                break
        open_cv.destroyWindow(self.caption)

    def __mouse_callback(self, event, x, y, flags, params):

        if event == open_cv.EVENT_LBUTTONDOWN:
            self.coordinates.append((x, y))
            self.click_count += 1

            if self.click_count >= 4:
                self.__handle_done()

            elif self.click_count > 1:
                self.__handle_click_progress()

        open_cv.imshow(self.caption, self.image)

    def __handle_click_progress(self):
        open_cv.line(self.image, self.coordinates[-2], self.coordinates[-1], (255, 0, 0), 1)

    def __handle_done(self):
        open_cv.line(self.image,
                     self.coordinates[2],
                     self.coordinates[3],
                     self.color,
                     1)
        open_cv.line(self.image,
                     self.coordinates[3],
                     self.coordinates[0],
                     self.color,
                     1)

        self.click_count = 0

        coordinates = np.array(self.coordinates)

        self.output.write("-\n          id: " + str(self.ids) + "\n          coordinates: [" +
                          "[" + str(self.coordinates[0][0]) + "," + str(self.coordinates[0][1]) + "]," +
                          "[" + str(self.coordinates[1][0]) + "," + str(self.coordinates[1][1]) + "]," +
                          "[" + str(self.coordinates[2][0]) + "," + str(self.coordinates[2][1]) + "]," +
                          "[" + str(self.coordinates[3][0]) + "," + str(self.coordinates[3][1]) + "]]\n")

        draw_contours(self.image, coordinates, str(self.ids + 1), COLOR_WHITE)

        for i in range(0, 4):
            self.coordinates.pop()

        self.ids += 1