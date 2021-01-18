
import cv2
import numpy as np
import object_tracker as tracker

# Util functions
def dist(a, b):
    return np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))

def get_angle(v1, v2):
    dx = v2[0] - v1[0]
    dy = v2[1] - v2[1]
    return np.arctan2(dy, dx) * 180 / np.pi

def norm_dist(v1, v2, sig=15):
    theta = get_angle(v1, v2)

    x = v1[0] + sig * np.cos(theta)
    y = v1[1] + sig * np.sin(theta)

    # print('check', dist((x, y), v1))
    # print("??", v1, "X:", (int(x),  "y", int(y)))
    return v1, (int(x), int(y))


# Lucas Kanade optical flow
def optical_flow(old_gray, frame_gray, p0):
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if st is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        return good_new, good_old
    else:
        return None, None


# Displays output for each frame
def draw_status(frame, mean_angle, kps):

    if 0 < mean_angle < 180:
        motion = 'forward'
    elif 180 < mean_angle < 360:
        motion = 'backward'
    else:
        motion = 'stopped'

    if kps < 300:
        motion = 'stopped'

    if motion == 'forward':
        cv2.rectangle(frame, (20, 5), (200, 35), (0, 255, 0), -1)
    else:
        cv2.rectangle(frame, (20, 5), (200, 35), (0, 0, 255), -1)

    cv2.putText(frame, 'Motion: ' + motion, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.rectangle(frame, (20, 45), (200, 75), (0, 255, 0), -1)
    cv2.putText(frame, 'Keypoints: ' + str(kps), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame


def get_features(gray, max_features_to_be_tracked):
    feature_detector = cv2.ORB_create(max_features_to_be_tracked)
    kp, descs = feature_detector.detectAndCompute(gray, None)
    return np.array([np.array([k.pt]).astype(np.float32,) for k in kp])


def flow_calculation(frame,
                     old_gray,
                     w,
                     h,
                     p0,
                     max_features_to_be_tracked,
                     frame_count):
    ### overal flow to update unseen cars and trackers which is located after 0.8*w
    overal_flow_x = []
    overal_flow_y = []

    old_gray
    color = np.random.randint(0, 255, (max_features_to_be_tracked, 3))
    frame_flow = frame[0:h, int(w / 2):w]
    frame_gray = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2GRAY)
    angles = [0]
    kps = 0
    good_new, good_old = optical_flow(old_gray, frame_gray, p0)

    if good_new is not None:
        min_distance = 0.8

        # Collect angle between keypoint points and draw vectors to frame
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            if min_distance < dist((a,b), (c, d)):
                kps += 1
                angles.append(get_angle((a,b), (c,d)))
                xx, yy = norm_dist((a+w/2,b), (c+w/2,d))
                cv2.arrowedLine(frame, (int(yy[0]), int(yy[1])), (int(xx[0]),
                                                                  int(xx[1])), color[i].tolist(), 1, tipLength=0.2)
                cv2.circle(frame, (int(yy[0]), int(yy[1])), 2, color[i].tolist(), -1)
                overal_flow_x.append(xx[1] - yy[1])
                overal_flow_y.append(xx[0] - yy[0])



    # Calculate the mean angle every n frames
    if frame_count % 1 == 0:
        current_angle = np.mean(angles)

    draw_status(frame, current_angle, kps)


    delta_x = tracker.average_list(overal_flow_x)
    delta_y = tracker.average_list(overal_flow_y)

    if kps < 300:
        delta_x, delta_y = 0, 0
    return frame_gray, good_new, delta_x, delta_y
