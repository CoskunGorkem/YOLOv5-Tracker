import cv2
import object_tracker as tracker

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyse_whole_video(source):
    cap = cv2.VideoCapture(source)

    average_blur_list = []
    while (cap.isOpened()):

        ok, frame = cap.read()
        if ok == True:
            blur_index = variance_of_laplacian(frame)
            average_blur_list.append(blur_index)
        else:
            break

    average_blur = tracker.average_list(average_blur_list)
    #print("average_blur", average_blur)
    return average_blur


