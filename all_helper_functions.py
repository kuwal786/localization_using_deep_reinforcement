from keras.preprocessing import image
import numpy as np


def get_all_ids(annotations):
    all_ids = []
    for i in range(len(annotations)):
        all_ids.append(get_ids_objects_from_annotation(annotations[i]))
    return all_ids


def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def get_all_images_pool(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def load_images_names_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    if data_set_name.startswith("aeroplane") | data_set_name.startswith("bird") | data_set_name.startswith("cow"):
        return [x.split(None, 1)[0] for x in image_names]
    else:
        return [x.strip('\n') for x in image_names]


def load_images_labels_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    images_names = f.readlines()
    images_names = [x.split(None, 1)[1] for x in images_names]
    images_names = [x.strip('\n') for x in images_names]
    return images_names


def mask_image_with_mean_background(mask_object_found, image):
    new_image = image
    size_image = np.shape(mask_object_found)
    for j in range(size_image[0]):
        for i in range(size_image[1]):
            if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 103.939
                    new_image[j, i, 1] = 116.779
                    new_image[j, i, 2] = 123.68
    return new_image






#metrics
import cv2


def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou


def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_masks, mask, array_classes_gt_objects, object_id, last_matrix, available_objects):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(array_classes_gt_objects)):
        if array_classes_gt_objects[k] == object_id:
            if available_objects[k] == 1:
                gt_mask = gt_masks[:, :, k]
                iou = calculate_iou(mask, gt_mask)
                results[k] = iou
            else:
                results[k] = -1
    max_result = max(results)
    ind = np.argmax(results)
    iou = last_matrix[ind]
    new_iou = max_result
    return iou, new_iou, results, ind






#parse_xml_annotations
import xml.etree.ElementTree as ET


def get_bb_of_gt_from_pascal_xml_annotation(xml_name, voc_path):
    string = voc_path + '/Annotations/' + xml_name + '.xml'
    tree = ET.parse(string)
    root = tree.getroot()
    names = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    for child in root:
        if child.tag == 'object':
            for child2 in child:
                if child2.tag == 'name':
                    names.append(child2.text)
                elif child2.tag == 'bndbox':
                    for child3 in child2:
                        if child3.tag == 'xmin':
                            x_min.append(child3.text)
                        elif child3.tag == 'xmax':
                            x_max.append(child3.text)
                        elif child3.tag == 'ymin':
                            y_min.append(child3.text)
                        elif child3.tag == 'ymax':
                            y_max.append(child3.text)
    category_and_bb = np.zeros([np.size(names), 5])
    for i in range(np.size(names)):
        category_and_bb[i][0] = get_id_of_class_name(names[i])
        category_and_bb[i][1] = x_min[i]
        category_and_bb[i][2] = x_max[i]
        category_and_bb[i][3] = y_min[i]
        category_and_bb[i][4] = y_max[i]
    return category_and_bb


def get_all_annotations(image_names, voc_path):
    annotations = []
    for i in range(np.size(image_names)):
        image_name = image_names[0][i]
        annotations.append(get_bb_of_gt_from_pascal_xml_annotation(image_name, voc_path))
    return annotations


def generate_bounding_box_from_annotation(annotation, image_shape):
    length_annotation = annotation.shape[0]
    masks = np.zeros([image_shape[0], image_shape[1], length_annotation])
    for i in range(0, length_annotation):
        masks[annotation[i, 3]:annotation[i, 4], annotation[i, 1]:annotation[i, 2], i] = 1
    return masks


def get_ids_objects_from_annotation(annotation):
    return annotation[:, 0]


def get_id_of_class_name (class_name):
    if class_name == 'aeroplane':
        return 1
    elif class_name == 'bicycle':
        return 2
    elif class_name == 'bird':
        return 3
    elif class_name == 'boat':
        return 4
    elif class_name == 'bottle':
        return 5
    elif class_name == 'bus':
        return 6
    elif class_name == 'car':
        return 7
    elif class_name == 'cat':
        return 8
    elif class_name == 'chair':
        return 9
    elif class_name == 'cow':
        return 10
    elif class_name == 'diningtable':
        return 11
    elif class_name == 'dog':
        return 12
    elif class_name == 'horse':
        return 13
    elif class_name == 'motorbike':
        return 14
    elif class_name == 'person':
        return 15
    elif class_name == 'pottedplant':
        return 16
    elif class_name == 'sheep':
        return 17
    elif class_name == 'sofa':
        return 18
    elif class_name == 'train':
        return 19
    elif class_name == 'tvmonitor':
        return 20








#visualizations


from PIL import Image, ImageDraw, ImageFont


path_font = "/usr/share/fonts/liberation/LiberationMono-Regular.ttf"
font = ImageFont.truetype(path_font, 24)


def string_for_action(action):
    if action == 0:
        return "START"
    if action == 1:
        return 'up-left'
    elif action == 2:
        return 'up-right'
    elif action == 3:
        return 'down-left'
    elif action == 4:
        return 'down-right'
    elif action == 5:
        return 'center'
    elif action == 6:
        return 'TRIGGER'


def draw_sequences(i, k, step, action, draw, region_image, background, path_testing_folder, iou, reward,
                   gt_mask, region_mask, image_name, save_boolean):
    mask = Image.fromarray(255 * gt_mask)
    mask_img = Image.fromarray(255 * region_mask)
    image_offset = (1000 * step, 70)
    text_offset = (1000 * step, 550)
    masked_image_offset = (1000 * step, 1400)
    mask_offset = (1000 * step, 700)
    action_string = string_for_action(action)
    footnote = 'action: ' + action_string + ' ' + 'reward: ' + str(reward) + ' Iou:' + str(iou)
    draw.text(text_offset, str(footnote), (0, 0, 0), font=font)
    img_for_paste = Image.fromarray(region_image)
    background.paste(img_for_paste, image_offset)
    background.paste(mask, mask_offset)
    background.paste(mask_img, masked_image_offset)
    file_name = path_testing_folder + '/' + image_name + str(i) + '_object_' + str(k) + '.png'
    if save_boolean == 1:
        background.save(file_name)
    return background


def draw_sequences_test(step, action, qval, draw, region_image, background, path_testing_folder,
                        region_mask, image_name, save_boolean):
    aux = np.asarray(region_image, np.uint8)
    img_offset = (1000 * step, 70)
    footnote_offset = (1000 * step, 550)
    q_predictions_offset = (1000 * step, 500)
    mask_img_offset = (1000 * step, 700)
    img_for_paste = Image.fromarray(aux)
    background.paste(img_for_paste, img_offset)
    mask_img = Image.fromarray(255 * region_mask)
    background.paste(mask_img, mask_img_offset)
    footnote = 'action: ' + str(action)
    q_val_predictions_text = str(qval)
    draw.text(footnote_offset, footnote, (0, 0, 0), font=font)
    draw.text(q_predictions_offset, q_val_predictions_text, (0, 0, 0), font=font)
    file_name = path_testing_folder + image_name + '.png'
    if save_boolean == 1:
        background.save(file_name)
    return background












