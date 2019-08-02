#coding=utf-8
import os
import argparse
import re
import numpy as np
import cv2
import glob
import json
import codecs
import pdb
from cutplate import four_point_transform, stitch
import sys
import random
from shutil import copyfile

"""
Crop only the hongkong plate
"""

#chars = "1234567890QWERTYUPASDFGHJKLZXCVBNM京鄂津湘冀粤晋桂蒙琼辽渝吉川黑贵沪云苏藏浙陕皖甘闽青赣宁鲁新豫警港澳使领学试挂".decode('utf8')
charsJson = "1234567890QWERTYUPASDFGHJKLZXCVBNM京鄂津湘冀粤晋桂蒙琼辽渝吉川黑贵沪云苏藏浙陕皖甘闽青赣宁鲁新豫警港澳使领学试挂".decode('utf8')
chars = "不京鄂津湘冀粤晋桂蒙琼辽渝吉川黑贵沪云苏藏浙陕皖甘闽青赣宁鲁新豫警港澳使领学试挂清模糊缺".decode('utf8')

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def construct_ocr_label(text, out_path):
    plate_ocr = text.strip()
    plate_ocr = plate_ocr.replace(' ', '')
    plate_ocr = plate_ocr.replace('-', '')
    plate_label = '|' + '|'.join(plate_ocr) + '|'
    line = out_path + ';' + plate_label + '\n'
    return line

def ocrlabel_to_jsonlabel(json_file, ocrtxt_file, plate_dir,corner_results,plate_type_results):
    """
    Format of the JSON
    return:
    """

    json_data ={}
    i=0
    for line in open(ocrtxt_file, 'r'):
        fpath, label= line.split(';')
        fname = os.path.basename(fpath)
        key = fname.replace("_stitched.jpg",'')
        # print fname
        # print '### Processing ', i, fname
        fpath = os.path.join(plate_dir, fname)
        if not os.path.exists(fpath):
            print fname, "does not exists!!!"
            continue
        plate = label.strip().decode('utf8')
        plate = plate.replace('|', '')
        # print plate
        if plate[0] not in charsJson:
            continue
        # print plate
        plate_anno =[]
        plate_anno.append(
            {"coordinates": [corner_results[key]],
             "text": plate,
             "plate_type": plate_type_results[key],}
        )
        i+=1
        if fname not in json_data:
            json_data[fname] = plate_anno
    with codecs.open(json_file, 'w') as f:
        json.dump(json_data, f,indent=4)

    print "OCR text label file converted to json file", json_file


def write_plate_image(img_path, json_path, out_dir, label_txt, lp_dir_single, lp_dir_double, stitched_dir, stitched_label_txt,stitched_json):
    """
    Crop the plate from the car image according to label results (json file),
    and write it to out_dir

    If there is multiple rows in a plate, (like hongkong plate),
    write multiple plate images into files.
    """
    corner = json.load(open(json_path))
    corner_results = {}
    plate_type_results = {}
    dst_folder = glob.glob(img_path + '/*.jpg')
    src_folderSingle = glob.glob(lp_dir_single + '/*.jpg')
    src_folderDouble = glob.glob(lp_dir_double + '/*.jpg')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fo = codecs.open(label_txt, "w", encoding='utf-8')
    fo1 = codecs.open(stitched_label_txt, "w", encoding='utf-8')

    hklp_singleList = []
    hklp_doubleList = []
    for i, image_name in enumerate(src_folderSingle):
        img = cv2.imdecode(np.fromfile(image_name,dtype=np.uint8),-1)
        if img is None:
            continue
        hklp_singleList.append((image_name,img))
    for i, image_name in enumerate(src_folderDouble):
        img = cv2.imdecode(np.fromfile(image_name,dtype=np.uint8),-1)
        if img is None:
            continue
        hklp_doubleList.append((image_name,img))

    for i, image_name in enumerate(dst_folder):
        bname = os.path.basename(image_name).decode('utf8')

        #pdb.set_trace()
        if bname not in corner:
            continue

        if not os.path.exists(image_name):
           continue  

        img = cv2.imdecode(np.fromfile(image_name,dtype=np.uint8),-1)
        #pdb.set_trace()
        if img is None:
            continue
        # if img.shape[1] > 3000 or img.shape[0] > 3000:
        #     continue
        try:
            for rr, det in enumerate(corner[bname]):
                if not det['text']:
                    continue
                first_char = det['text'][0]
                if first_char in chars:
                    continue

                print('### {}: Processing {}'.format(i, bname.encode('utf8')))
                plate_type = 'single_row'
                if 'plate_type' in det:
                    plate_type = det['plate_type']

                corner_pts = det['coordinates']
                if len(corner_pts) < 8:
                    print('{}: {} length of pts shorter than 8'.format(i, bname.encode('utf8')))
                    continue
                corner_pts = corner_pts[0:8]
                plate_img, M, maxWidth, maxHeight = four_point_transform(img, corner_pts)
                
                #pdb.set_trace()
                if plate_type == "single_row":
                    src_img = random.choice(hklp_singleList)
                    dst_img = img
                    plate_chars,_ = os.path.basename(src_img[0]).split('_')
                    cv2.imshow('',src_img[1])
                    cv2.waitKey(0)
                    print plate_chars
                    stitched_img,corner_pts = stitch(src_img[1], dst_img, corner_pts)
                    cv2.imshow('',stitched_img)
                    cv2.waitKey(0)
                    key = bname.replace('.jpg', '')
                    stitched_path = os.path.join(stitched_dir,  bname.replace('.jpg', '_stitched.jpg'))
                    cv2.imwrite(stitched_path, stitched_img)
                    line = construct_ocr_label(plate_chars, stitched_path)
                    print line.encode('utf8')
                    fo1.write("%s" % line)
                    

                    out_path = os.path.join(out_dir,  bname.replace('.jpg', '_plate.jpg'))
                    cv2.imwrite(out_path, plate_img)
                    line = construct_ocr_label(det['text'], out_path)
                    print line.encode('utf8')
                    fo.write("%s" % line)
                    # corner_results.append(corner_pts)
                    corner_results[key] = corner_pts
                    # plate_type_results.append(plate_type)
                    plate_type_results[key] = plate_type
                    
                if plate_type == "double_row":
                    src_img = random.choice(hklp_doubleList)
                    dst_img = img
                    stitched_img, corner_pts = stitch(src_img[1], dst_img, corner_pts)
                    key = bname.replace('.jpg', '')
                    stitched_path = os.path.join(stitched_dir,  bname.replace('.jpg', '_stitched.jpg'))
                    cv2.imwrite(stitched_path, stitched_img)
                    plate_chars,_ = os.path.basename(src_img[0]).split('_')
                    line = construct_ocr_label(plate_chars, stitched_path) 
                    
                    line.encode('utf8')
                    fo1.write("%s" % line)

                    out_path1 = os.path.join(out_dir, bname.replace('.jpg', '_plate1.jpg'))
                    out_path2 = os.path.join(out_dir, bname.replace('.jpg', '_plate2.jpg'))
                    height, width = plate_img.shape[0:2]
                    plate_img1 = plate_img[:height/2,:]
                    plate_img2 = plate_img[height/2:,:]

                    line1 = construct_ocr_label(det['text'][:2], out_path1)
                    line2 = construct_ocr_label(det['text'][2:], out_path2)
                    cv2.imwrite(out_path1, plate_img1)
                    cv2.imwrite(out_path2, plate_img2)

                    print line1.encode('utf8')
                    print line2.encode('utf8')
                    fo.write("%s" % line1)
                    fo.write("%s" % line2)
                    # corner_results.append(corner_pts)
                    # plate_type_results.append(plate_type)
                    corner_results[key] = corner_pts
                    plate_type_results[key] = plate_type
        except:
            continue
    fo.close()
    fo1.close()
    ocrlabel_to_jsonlabel(stitched_json, stitched_label_txt ,stitched_dir,corner_results, plate_type_results)

def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--lp_dir_single', default=sys.path[0] + '/hklp_single/',
                        type=str, help='Input Single Hong Kong License Plate image dir')
    parser.add_argument('--lp_dir_double', default=sys.path[0] + '/hklp_double/',
                        type=str, help='Input Double Hong Kong License Plate image dir')
    parser.add_argument('--img_dir', default=sys.path[0] + '/car_crop_20190505/',
                        type=str, help='Input test image dir')
    parser.add_argument('--label_json', default=sys.path[0] + '/20190505_HK_Double_Plates.json',
                        type=str, help='Plate labeling results in json')
    parser.add_argument('--plate_dir', default=sys.path[0] + '/plates/',
                        type=str, help='Output plate image dir')
    parser.add_argument('--stitched_dir', default=sys.path[0] + '/stitched/',
                        type=str, help='Output plate image dir')
    parser.add_argument('--label_txt', default=sys.path[0] + '/label.txt',
                        type=str, help='Output OCR label txt')
    parser.add_argument('--stitched_label_txt', default=sys.path[0] + '/stitched_label.txt',
                        type=str, help='Output Stitched OCR label txt')
    parser.add_argument('--stitched_json', default=sys.path[0] + '/stitched.json',
                        type=str, help='Output Stitched json file')
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    try:
        shutil.rmtree(args.plate_dir)
    except:
        pass
    try:
        shutil.rmtree(args.stitched_dir)
    except:
        pass
    try:
        os.mkdir(args.plate_dir)
    except:
        pass
    try:
        os.mkdir(args.stitched_dir)
    except:
        pass
    write_plate_image(args.img_dir, args.label_json, args.plate_dir, args.label_txt,args.lp_dir_single, args.lp_dir_double,args.stitched_dir, args.stitched_label_txt,args.stitched_json)

