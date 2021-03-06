import os
import json
import argparse

import cv2
import numpy as np


file_dir = os.path.dirname(__file__)


def parser():
    parser = argparse.ArgumentParser(description='Color-Based Video Cropper')
    parser.add_argument('--input_video', type=str, required=True,
                        help='path of the input video')
    parser.add_argument('--output_video', type=str, required=True,
                        help='path of the output video')
    parser.add_argument('--output_size', type=str, default="auto",
                        help='output_size of the input video, widthxheight, like 720x1080')
    parser.add_argument('--pad_mode', type=str, default="inside",
                        help='inside or outside, inside means padding, outside means cropping')
    parser.add_argument('--edge_color', type=str, default="auto",
                        help='padding color')
    parser.add_argument('--color', type=str, default=os.path.join(file_dir, '../data/color.json'),
                        help='path of color.json')
    args = parser.parse_args()
    return args


def bgr2hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = hsv[:,:,0]*2
    hsv[:,:,1] = hsv[:,:,1]/255
    hsv[:,:,2] = hsv[:,:,2]/255
    return hsv


def color_mask(image, color_range):
    flag = np.zeros(image.shape[:2])
    for i in range(len(color_range['h'])):
        lower = np.array([color_range['h'][0][0], color_range['s'][0][0], color_range['v'][0][0]], dtype=np.float32)
        upper = np.array([color_range['h'][0][1], color_range['s'][0][1], color_range['v'][0][1]], dtype=np.float32)
        hsv = bgr2hsv(image)
        flag += cv2.inRange(hsv, lower, upper)
    return flag


def flag_range(vec, thr=240):
    res = []
    s = 0
    n = len(vec)
    while s < n:
        while s < n and vec[s] < thr:
            s += 1
        e = s
        while e < n and vec[e] >= thr:
            e += 1
        if e > s:
            res.append((s, e))
        s = e
    return res


def content_range(mask, axis, min_length, thr=240):
    vecs = mask.mean(axis=axis)
    flags = flag_range(vecs, thr=thr)
    num = len(flags)

    content = []
    end_idx = len(vecs)
    start_idx = 0
    if num == 0:
        content.append((start_idx, end_idx))
        return content

    for i in range(num):
        s, e = flags[i]
        if s != start_idx and s - start_idx >= min_length:
            content.append((start_idx, s))
        start_idx = e
        if i == num - 1 and e != end_idx and end_idx - e >= min_length:
            content.append((e, end_idx))
    return content


def cut_image(img, color_range, min_length_ratio=0.2):
    res = []
    mask = color_mask(img, color_range)
    min_length0 = mask.shape[0] * min_length_ratio
    min_length1 = mask.shape[1] * min_length_ratio
    content = content_range(mask, 1, min_length0)
    for item in content:
        cont_mask = mask[item[0]:item[1], :]
        cont = content_range(cont_mask, 0, min_length1)
        for col in cont:
            res.append([(item[0], item[1]), (col[0], col[1])])
    return res


def get_hsv_name(point, color_ranges):
    point_name = None
    for name, color_range in color_ranges.items():
        lower = np.array([color_range['h'][0][0], color_range['s'][0][0], color_range['v'][0][0]], dtype=np.float32)
        upper = np.array([color_range['h'][0][1], color_range['s'][0][1], color_range['v'][0][1]], dtype=np.float32)
        hsv = bgr2hsv(point[np.newaxis, np.newaxis, :])
        flag = cv2.inRange(hsv, lower, upper)
        if flag[0]:
            point_name = name
            break
    return point_name


def get_edge_color(args, img_arr):
    with open(args.color, 'r') as f:
        color_ranges = json.load(f)
    edge_color = args.edge_color
    if edge_color == "auto":
        names = []
        h, w, _ = img_arr.shape
        for point in [[0, 0], [h - 1, 0], [0, w - 1], [h - 1, w - 1]]:
            name = get_hsv_name(img_arr[point[0], point[1]], color_ranges)
            names.append(name)
        name = list(set(names))
        if len(name) > 1:
            raise Exception("Colors at four corners are different, failed to get color automatically.Please specify a color name.")
        else:
            print ("Using color-{}".format(name[0]))
            tmp_color = color_ranges.get(name[0])
            assert tmp_color is not None, 'no color named {}'.format(name)
            edge_color = tmp_color
    else:
        print ("Using specified color-{}".format(edge_color))
        tmp_color = color_ranges.get(edge_color)
        assert tmp_color is not None, 'no color named {}'.format(edge_color)
        edge_color = tmp_color
    print ("edge_color range: {}".format(edge_color))
    return edge_color


def save_video(cap, save_path, img_arr, size, fps, r, args):
    print ("output size is {}x{}".format(size[0], size[1]))
    print ("output path is {}".format(save_path))
    c_h, c_w = img_arr.shape[:2]
    (h0, h1), (w0, w1) = r
    out_w, out_h = size
    out = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, size
    )
    if args.output_size == "auto" or args.pad_mode != "outside":
        ratio = max(out_w / c_w, out_h / c_h)
        r_w, r_h = int(c_w*ratio), int(c_h*ratio)
        r_img_arr = cv2.resize(img_arr, (r_w, r_h))
        start_w = int(r_w / 2 - out_w / 2)
        end_w = start_w + out_w
        start_h = int(r_h / 2 - out_h / 2)
        end_h = start_h + out_h
        output_arr = r_img_arr[start_h:end_h, start_w:end_w, :]
        out.write(output_arr)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = frame[h0:h1, w0:w1, :]
                r_img_arr = cv2.resize(frame, (r_w, r_h))
                start_w = int(r_w  / 2- out_w / 2)
                end_w = start_w + out_w
                start_h = int(r_h / 2 - out_h / 2)
                end_h = start_h + out_h
                output_arr = r_img_arr[start_h:end_h, start_w:end_w, :]
                out.write(output_arr)
            else:
                break
    else:
        ratio = min(out_w / c_w, out_h / c_h)
        r_w, r_h = int(c_w*ratio), int(c_h*ratio)
        empty_frame = np.zeros((out_h, out_w, img_arr.shape[2]), dtype=np.uint8)
        r_img_arr = cv2.resize(img_arr, (r_w, r_h))
        start_w = int(out_w / 2 - r_w / 2)
        end_w = start_w + r_w
        start_h = int(out_h / 2 - r_h/ 2)
        end_h = start_h + r_h
        empty_frame[start_h:end_h, start_w:end_w, :] = r_img_arr
        out.write(empty_frame)
        while True:
            ret, frame = cap.read()
            if ret:
                empty_frame = np.zeros((out_h, out_w, img_arr.shape[2]), dtype=np.uint8)
                frame = frame[h0:h1, w0:w1, :]
                r_img_arr = cv2.resize(frame, (r_w, r_h))
                start_w = int(out_w / 2 - r_w / 2)
                end_w = start_w + r_w
                start_h = int(out_h / 2 - r_h/ 2)
                end_h = start_h + r_h
                empty_frame[start_h:end_h, start_w:end_w, :] = r_img_arr
                out.write(empty_frame)
            else:
                break
    out.release()


if __name__ == '__main__':
    args = parser()
    # crop by first frame
    cap = cv2.VideoCapture(args.input_video)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    ret, img_arr = cap.read()
    h, w, _ = img_arr.shape
    if not ret:
        raise Exception("Failed to read video, please make sure {} is right".format(args.input_video))
    edge_color = get_edge_color(args, img_arr)
    ranges = cut_image(img_arr, edge_color, min_length_ratio=0.15)
    flag = False
    for r in ranges:
        (h0, h1), (w0, w1) = r
        if h0 < h/2 and h1 > h/2 and w0 < w/2 and w1 > w/2:
            flag = True
            break

    save_path = args.output_video
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    if not flag:
        print ("image not at center")
    (h0, h1), (w0, w1) = r
    crop_image = img_arr[h0:h1, w0:w1, :]
    c_h, c_w = crop_image.shape[:2]
    if args.output_size != "auto":
        out_w, out_h = list(map(int, args.output_size.split("x")))
        save_video(cap, save_path, crop_image, (out_w, out_h), fps, r, args)
    else:
        save_video(cap, save_path, crop_image, (c_w, c_h), fps, r, args)
    cap.release()
    print ("Done")