import cv2
import math
from copy import deepcopy


def average_list(lst):
    try:
        return sum(lst) / len(lst)
    except:
        return (0)

def cent_distance_calc(tuple1, tuple2):
    x1,y1 = tuple1
    x2,y2 = tuple2
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def det_box_area(box):
    '''
    Calculates the area of bounding box to filter out some of them
    '''
    return ((box[1]-box[0])+1)*((box[3]-box[2]))
    
    
def intersection_over_union(boxA, boxB):
    
    '''
    box Structure Xmin, Xmax, Ymin, Ymax
    Function to calculaate intersection over union to improve tracker algorithm
    :param boxA: bounding box
    :param boxB: tracker box
    :return: intersection of 2 boxes
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = det_box_area(boxA)
    boxBArea =det_box_area(boxB)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
    
def box_ratio(box):
    try:
        return (box[1]-box[0])/(box[3]-box[2])
    except:
        return 0
    

def tracker_assign_update(frame,
                   t_list,
                   t_coordinates,
                   t_io,
                   t_confidence,
                   box_confidence,
                   center,
                   class_name,
                   plus_index,
                   box,
                   tl1,
                   tl2,
                   result_list,
                   distance_threshold,
                   timestamp,
                   ):

        '''
        ######################### REPORT FILE ########################
        '''
        dict = {}
        if tl1 != tl2:
            dict['timeSinceStartMs'] = timestamp * 1000
            dict['object_id'] = len(t_list) + plus_index
            dict['type'] = class_name
            result_list.append(dict)
        
        tl1 = len(t_list)
        tmp_list = deepcopy(t_list)
        tmp_coor = deepcopy(t_coordinates)
        tmp_io = deepcopy(t_io)
        tmp_confidence = deepcopy(t_confidence)

        distances = [cent_distance_calc(i, center) for i in tmp_list]
        min_distance = min(distances)
        min_distance_index = distances.index(min_distance)
        max_distance = max(distances)

        bb_ratio = box_ratio(box)


        '''
        #########################TRACKER BOX DEBUG VISUALS#########################
        '''
        cv2.rectangle(frame, (int(box[0]), int(box[3] - 110)), (int(box[0] + 90), int(box[3] - 90)), (0, 255, 0), -1)
        cv2.putText(frame, ("CS: " + str(box_confidence)[:4]),
                    (int(box[0]), int(box[3] - 95)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(frame, (int(box[0]), int(box[3] - 80)), (int(box[0] + 90), int(box[3] - 60)), (0, 255, 0), -1)
        cv2.putText(frame, ("BB_R: " + str(bb_ratio)[:4]),
                    (int(box[0]), int(box[3] - 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(frame, (int(box[0]), int(box[3] - 50)), (int(box[0] + 90), int(box[3] - 30)), (0, 255, 0), -1)
        cv2.putText(frame, ("IOU: " + str(intersection_over_union(box, tmp_coor[min_distance_index]))[:4]),
                    (int(box[0]), int(box[3] - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        '''
        ############################################################################
        '''


        if min_distance < distance_threshold and 0.50 < intersection_over_union(box,tmp_coor[min_distance_index]):
            tmp_list[min_distance_index] = center
            t_list = tmp_list

            tmp_coor[min_distance_index] = box
            t_coordinates = tmp_coor

            tmp_io[min_distance_index] = intersection_over_union(box, tmp_coor[min_distance_index])
            t_io = tmp_io

            tmp_confidence[min_distance_index] = box_confidence

            cv2.rectangle(frame, (int(box[0]), int(box[3] - 20)), (int(box[0] + 90), int(box[3])), (0, 255, 0), -1)
            cv2.putText(frame, "TRACKING..", (int(box[0]), int(box[3] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


            
        elif distance_threshold * 2 < max_distance and \
                intersection_over_union(box,tmp_coor[min_distance_index]) < 0.25 \
                and 0.9 < bb_ratio:

            #ASSIGN NEW TRACKER CENTERS
            tmp_list.append(center)
            t_list = tmp_list

            #ASSIGN NEW TRACKER COORDINATES
            tmp_coor.append(box)
            t_coordinates = tmp_coor

            #ASSIGN NEW TRACKER IOU
            tmp_io.append(intersection_over_union(box, tmp_coor[min_distance_index]))
            t_io = tmp_io

            #ASSIGN NEW TRACKER CONFIDENCE SCORE
            tmp_confidence.append(box_confidence)
            t_confidence = tmp_confidence



            cv2.rectangle(frame, (int(box[0]), int(box[3] - 20)), (int(box[0] + 90), int(box[3])), (0, 0, 255), -1)
            cv2.putText(frame, "NEW", (int(box[0]), int(box[3] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


            if t_list[0] == (0,0):
                print("TRACKER INITIALIZED")
                t_list.pop(0)
                t_coordinates.pop(0)
                t_io.pop(0)
                t_confidence.pop(0)

                tl1 = tl1 - 1

        else:
            # cv2.rectangle(frame, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (0, 0, 255), 2)
            cv2.rectangle(frame, (int(box[0]), int(box[3] - 20)), (int(box[0] + 90), int(box[3])), (0, 0, 255), -1)
            cv2.putText(frame, "Untracked", (int(box[0]), int(box[3] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                
        
        
        tl2 = len(t_list)

        return t_list, t_coordinates, t_io, t_confidence, tl1, tl2, result_list




def tracker_cleaner(T_LIST,
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
                    frame):
    delete_list = []
    if T_LIST[0] != (0, 0):
        for iou_index, iou in enumerate(t_io_old):
            '''
            if object is:
            1- visible
            2- IOU is same
            3- Tracker center point is same
            Append that object in punishment list
            If object will be there for max_punish_occurance times: delete the object from all lists.
            '''
            max_penalty = 20
            if T_COORDINATES[iou_index][0] < w:
                if iou == T_IO[iou_index]:
                    if t_old_center[iou_index] == T_LIST[iou_index]:
                        if T_PENALTY.count(iou_index) < max_penalty:
                            T_PENALTY.append(iou_index)
                            T_COORDINATES[iou_index] = (T_COORDINATES[iou_index][0] + (x_delta), T_COORDINATES[iou_index][1] + x_delta,
                                                            T_COORDINATES[iou_index][2] + y_delta, T_COORDINATES[iou_index][3] + y_delta)
                            T_LIST[iou_index] = ((T_COORDINATES[iou_index][0] + T_COORDINATES[iou_index][1]) / 2 + x_delta, (T_COORDINATES[iou_index][2] + T_COORDINATES[iou_index][3]) / 2 + y_delta)
                        else:
                            if iou_index not in delete_list:
                                if w * 0.9 < T_COORDINATES[iou_index][0]:
                                    pass
                                else:
                                    delete_list.append(iou_index)

        if delete_list and len(T_LIST) == len(RESULT_LIST):
            del T_COORDINATES[delete_list[0]]
            del T_LIST[delete_list[0]]
            del T_IO[delete_list[0]]
            del T_CONFIDENCE[delete_list[0]]
            del RESULT_LIST[delete_list[0]]
            T_PENALTY = list(filter((delete_list[0]).__ne__, T_PENALTY))

            if len(T_LIST) == 0:
                T_LIST.append((0, 0))
                T_COORDINATES.append((0,0,0,0))
                T_IO.append(0)
                T_CONFIDENCE.append(0)

            cv2.rectangle(frame, (20, 205), (200, 235), (0, 0, 255), -1)
            cv2.putText(frame, ("Deleted Park ID: " + str(iou_index+1)), (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 1)

    return T_PENALTY
#
#
#
# def sort_report(list):
#     '''
#     Function that takes a list and sort it the values with respect to their x coordinates
#     It is used just to see results easier
#     It does not have any effect to performance of the tracker
#     '''
#     l = len(list)
#     for i in range(0, l):
#         for j in range(0, l-i-1):
#             if (list[j]['timeSinceStartMs'] > list[j + 1]['timeSinceStartMs']):
#                 tempo = list[j]
#                 list[j]= list[j+1]
#                 list[j + 1] = tempo
#     return list


#
# def sort_list(list, index):
#     '''
#     Function that takes a list and sort it the values with respect to their index coordinates
#     It is used just to see results easier, to order car IDs from lower to higher.
#     It does not have any effect to performance of the tracker
#     '''
#
#     l = len(list)
#     for i in range(0, l):
#         for j in range(0, l-i-1):
#             if (list[j][index] > list[j + 1][index]):
#                 tempo = list[j]
#                 list[j]= list[j + 1]
#                 list[j + 1] = tempo
#     return list



