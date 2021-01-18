import os
import sys
import cv2
import flow
import time
import torch
import report
import argparse
import blur_detection as blur
import image_processing as ip
import object_tracker as tracker
import torch.backends.cudnn as cudnn

sys.path.append('..')

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging


def detect():
    source, weights, view_img, save_txt, imgsz, video_output = opt.source, opt.weights, \
                                                  opt.view_img, opt.save_txt, opt.img_size, opt.video_output

    print("AVERAGE BLUR" , blur.analyse_whole_video(source))
    print("=====VIDEO OUTPUT: ", video_output, " =============")
    if blur.analyse_whole_video(source) < 100:
        print("==========!!Video Quality is not good!!==========")
        print(source)
        print("==========Video is passed==========")
        
    else:
        '''
        ####################CAMERA & OUTPUT#######################
        '''
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        '''
        #######################DIRECTORIES########################
        '''
        if video_output:
            save_dir = opt.project
            source_name = source.split("/")[-1]
            os.makedirs(save_dir, exist_ok=True)
            result_video_path = os.path.join(save_dir, source_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        '''
        ######################FLOW PARAMETERS#####################
        '''
        min_features_to_be_tracked = 350
        max_features_to_be_tracked = 1000
        ret, old_frame = cap.read()
        old_frame = old_frame[0:h, int(w / 2):w]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = flow.get_features(old_gray, max_features_to_be_tracked)
        frame_count = 0
        '''
        ####################TRACKER PARAMETERS####################
        '''
        TL_1 = 0
        TL_2 = 0
        T_LIST = [(0,0)]
        T_COORDINATES = [(0,0,0,0)]
        T_IO = [0]
        T_CONFIDENCE = [0]
        T_THRESHOLD = 50
        B_AREA = 35000
        RESULT_LIST = []
        PLUS_INDEX = 0
        TIMESTAMP = 0
        T_PENALTY = []
        '''
        ##################INITIALIZE THE MODEL#####################
        '''
        # Initialize ===============================================
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model ===============================================
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        cudnn.benchmark = True
        names = model.module.names if hasattr(model, 'module') else model.names
        '''
        ####################ANALYSE VIDEO FRAME BY FRAME##############################
        '''
        while True:
            start_time = time.time()
            ok, frame = cap.read()

            if ok == False:
                if 2 < len(RESULT_LIST):
                    report.writer(RESULT_LIST,
                                  TIMESTAMP,
                                  source,
                                  )
                else:
                    print("Result list is empty")

            '''
            ####################FOCUS ANALYSIS####################
            '''
            blur_index = blur.variance_of_laplacian(frame)
            '''
            ####################FLOW CALCULATIONS####################
            '''
            frame_gray, good_new, x_delta, y_delta = flow.flow_calculation(frame,
                             old_gray,
                             w,
                             h,
                             p0,
                             max_features_to_be_tracked,
                             frame_count)
            '''
            #######################################################
            '''
            image = ip.letterbox(frame, new_shape=imgsz)
            image = ip.convert_image(image,device, half)
            pred = model(image, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            '''
            #################TRACKER ASSIGN & UPDATE###############
            '''
            t_io_old = T_IO
            t_old_center = T_LIST
            # detections per image
            for i, det in enumerate(pred):

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(image.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    class_name = cls.item()
                    class_name = names[int(class_name)]
                    confidence_score = conf.item()
                    x_min = xyxy[0].tolist()
                    y_min = xyxy[1].tolist()
                    x_max = xyxy[2].tolist()
                    y_max = xyxy[3].tolist()
                    center_point = (int((x_min+x_max)/2), int((y_min+y_max)/2))
                    box = x_min, x_max, y_min, y_max
                    area = tracker.det_box_area(box)

                    if 0 < area < B_AREA and w/2 < center_point[0] < w*0.8:
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

                    elif B_AREA < area and w/2 < center_point[0]:
                        if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                            T_LIST, T_COORDINATES, T_IO, T_CONFIDENCE, TL_1, TL_2, RESULT_LIST = \
                                tracker.tracker_assign_update(frame = frame,
                                                              t_list = T_LIST,
                                                              t_coordinates = T_COORDINATES,
                                                              t_io = T_IO,
                                                              t_confidence = T_CONFIDENCE,
                                                              box_confidence = confidence_score,
                                                              center = center_point,
                                                              class_name = class_name,
                                                              plus_index = PLUS_INDEX,
                                                              box = box,
                                                              tl1 = TL_1,
                                                              tl2 = TL_2,
                                                              result_list = RESULT_LIST,
                                                              distance_threshold = T_THRESHOLD,
                                                              timestamp = TIMESTAMP
                                                              )


                    else:
                        pass

            '''
            ##################TRACKER ASSIGN UPDATE ENDS##########################
            '''

            if T_LIST[0] != (0, 0):
                for tracker_index, box in enumerate(T_COORDINATES):
                    '''
                    Update unseen bounding boxes with optical flow
                    '''
                    if tracker.box_ratio(box) < 0.75 or w * 0.82 < box[0]:
                        T_COORDINATES[tracker_index] = (box[0] + 2 * (x_delta), box[1] + (2 * x_delta),
                                                        box[2] + y_delta, box[3] + y_delta)
                        T_LIST[tracker_index] = ((box[0] + box[1]) / 2 + (2 * x_delta), (box[2] + box[3]) / 2 + y_delta)
                    '''
                    Draw bounding boxes
                    '''
                    # Draw Bounding Boxes
                    cv2.rectangle(frame, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (0, 255, 0), 1)
                    # Draw the label background filled box
                    cv2.rectangle(frame, (int(box[0]), int(box[2] + 20)),
                                  (int(box[0] + 90), int(box[2])), (0, 255, 0), -1)
                    # Write the Park id
                    cv2.putText(frame, ("Park ID:" + str(tracker_index+1)), (int(box[0]), int(box[2] + 17)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                1)

            '''
            #######################DELETE FALSE TRACKERS######################
            #######################Update Result List######################
            '''
            T_PENALTY = tracker.tracker_cleaner(T_LIST,
                                               T_COORDINATES,
                                               T_IO,
                                               T_CONFIDENCE,
                                               T_PENALTY,
                                               RESULT_LIST,
                                               t_io_old,
                                               t_old_center,
                                               w,
                                               x_delta,
                                               y_delta,
                                               frame)
            '''
            #######################Info Panel######################
            '''

            cv2.rectangle(frame, (20, 85), (200, 115), (0, 255, 0), -1)
            if 0<len(T_LIST):
                if T_LIST[0] == (0,0):
                    parked_cars = '0'
                else:
                    parked_cars = str(len(T_COORDINATES))

            cv2.putText(frame, 'Occupied: ' + parked_cars,
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)


            cv2.rectangle(frame, (20, 125), (200, 155), (0, 255, 0), -1)
            cv2.putText(frame, 'x_d:' + str(x_delta)[:3] + ' y_d:' + str(y_delta)[:3], (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 1)

            #Write the blurry index on each frame
            if blur_index < 120:
                color = (0, 0, 255)
            elif 120 <= blur_index <= 250:
                color = (0,255,255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (20, 165), (200, 195), color, -1)
            cv2.putText(frame, ("Quality: " + str(round(blur_index,1))), (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 1)

            '''
            #######################################################
            '''
            end_time = time.time()
            inference_time = end_time - start_time
            TIMESTAMP += inference_time
            # print("TIMESTAMP", TIMESTAMP)
            print("Inference Time: ", inference_time)
            if video_output:
                out.write(frame)
            cv2.imshow('object detection', cv2.resize(frame, (720, 400)))

            # Update Frame and Keypoints
            old_gray = frame_gray.copy()
            if min_features_to_be_tracked < len(good_new):
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = flow.get_features(old_gray, max_features_to_be_tracked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test_video/test.mp4', help='source')
    parser.add_argument('--video_output', default=True, type=bool, help='uses the history file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='output_videos', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['../yolov5s.pt', '../yolov5m.pt', '../yolov5l.pt', '../yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
