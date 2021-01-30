import os
import sys
import cv2
import time
import torch
import argparse
import image_processing as ip
import object_tracker as tracker
import torch.backends.cudnn as cudnn
sys.path.append('..')
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging



def detect():
    source, weights, view_img, save_txt, imgsz, video_output, map = opt.source, opt.weights, \
                                                  opt.view_img, opt.save_txt, opt.img_size, opt.video_output, opt.map

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

        image = ip.letterbox(frame, new_shape=imgsz)
        image = ip.convert_image(image,device, half)
        pred = model(image, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


        # detections per image
        for i, det in enumerate(pred):

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):

                class_name = cls.item()
                class_name = names[int(class_name)]
                confidence_score = conf.item()
                x_min = int(xyxy[0].tolist())
                y_min = int(xyxy[1].tolist())
                x_max = int(xyxy[2].tolist())
                y_max = int(xyxy[3].tolist())
                box = x_min, x_max, y_min, y_max
                area = tracker.det_box_area(box)

                kernel_width = (w // 20) | 1
                kernel_height = (h // 20) | 1

                area = frame[y_min:y_max, x_min:x_max]
                blured_area = cv2.GaussianBlur(area, (kernel_width, kernel_height), 0)

                frame[y_min:y_max, x_min:x_max] = blured_area
                #
                # cv2.rectangle(frame, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (0, 255, 0), 1)
                # cv2.rectangle(frame, (int(box[0]), int(box[2] + 20)),
                #               (int(box[0] + 90), int(box[2])), (0, 255, 0), -1)
                # cv2.putText(frame, class_name, (int(box[0]), int(box[2] + 17)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                #             1)



        '''
        #######################################################
        '''
        end_time = time.time()
        inference_time = end_time - start_time
        # print("TIMESTAMP", TIMESTAMP)
        print("Inference Time: ", inference_time)
        if video_output:
            out.write(frame)
        cv2.imshow('object detection', cv2.resize(frame, (720, 400)))
        # Update Frame and Keypoints
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../anonymisation.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=0, help='source')
    parser.add_argument('--map', default=False, type=bool, help='uses the history file')
    parser.add_argument('--video_output', default=False, type=bool, help='uses the history file')
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
    parser.add_argument('--project', default='anonymised_output_videos', help='save results to project/name')
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
