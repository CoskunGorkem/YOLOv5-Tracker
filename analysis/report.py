import os
import sys
import json

from bson import json_util
from pymediainfo import MediaInfo


def sort_report(list):
    '''
    Function that takes a list and sort it the values with respect to their x coordinates
    It is used just to see results easier
    It does not have any effect to performance of the tracker
    '''
    l = len(list)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (list[j]['timeSinceStartMs'] > list[j + 1]['timeSinceStartMs']):
                tempo = list[j]
                list[j]= list[j+1]
                list[j + 1] = tempo
    return list


def writer(data,
           timestamp,
           video):
    media_info = MediaInfo.parse(video)
    duration_in_ms = media_info.tracks[0].duration
    current_path = os.getcwd()
    print("Full Video Duration:", duration_in_ms, "Model executed in", round(timestamp * 1000), "ms")
    for index, i in enumerate(data):
        i['timeSinceStartMs'] = round(i['timeSinceStartMs'] * (duration_in_ms / (timestamp * 1000))) + 2000


    # order the report file based on the timeSinceStartMs attribute
    data = sort_report(data)

    path_list = video.split("/")
    path_list_last = path_list[-1].split(".")

    report_name = path_list_last[0] + ".json"
    reports_folder_dir = os.path.join(current_path, 'reports')

    os.makedirs(reports_folder_dir, exist_ok=True)
    report_file_dir = os.path.join(reports_folder_dir, report_name)
    print("Report file is created")
    
    with open(report_file_dir, 'w') as outfile:
        json.dump(data, outfile, default=json_util.default)
    sys.exit("Label Has Recorded")



