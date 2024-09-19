import cv2 as cv

def show_image(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()