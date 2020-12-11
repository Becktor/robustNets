from __future__ import print_function

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision, noise=0, label="", plot=False):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([1.], precision, [0.]))

    if plot:
        # Plot pres vs recall
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        label_str = "Buoy" if label == 0 else "Boat"
        if noise <= 0.0:
            ax.set_title("Precision-Recall curve of {}".format(label_str, noise), loc='center', y=1.07)
        else:
            ax.set_title("Precision-Recall curve of {} with noise {}".format(label_str, noise), loc='center', y=1.07)

        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        minval = np.min(mpre[np.nonzero(mpre)])
        ax.set_ylim([0.88, 1.01])
        ax.set_xlim([0.0, 1.0])
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
        ax.plot(mrec, mpre)
        fig.tight_layout()
        fig.savefig("../figures/l_{}_n_{}_ap.png".format(label_str, noise))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_detections(dataset, model, score_threshold=0.05, max_detections=100, noise_level=0.0):
    """ Get the detections from the network using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the network.
        network           : The network to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        noise_level       : Amount of gaussian noise added
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]
    all_annotations = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]
    model.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            annotation = data['annot'].numpy()
            img = data['img']

            for label in range(dataset.num_classes()):
                all_annotations[index][label] = annotation[annotation[:, 4] == label, :4].copy()
            # if noise_level > 0:
            #   img = img + torch.empty(img.shape).normal_(mean=0, std=noise_level).numpy()

            # run network
            scores, labels, boxes = model(img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            #boxes /= scale
            debug = False

            if debug:
                import matplotlib.pyplot as plt
                import cv2
                img2 = img.numpy()
                bs = zip(boxes, scores)
                for v, s in bs:
                    if s > 0.3:
                        img2 = cv2.rectangle(img2, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])),color=(0,0,1), thickness=2)
                plt.imshow(img2)
                plt.show()


            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections, all_annotations


def get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations

        img = generator.load_image(i)
        annotations = generator.load_annotations(i, img.shape)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        noise=0,
        save_path=None,
        detections=None,
        annotations=None,
        f=None,
        plot=False
):
    """ Evaluate a given dataset using a given network.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The network to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations
    if not detections:
        all_detections, all_annotations = get_detections(generator, model, score_threshold=score_threshold,
                                                         max_detections=max_detections)
        # all_annotations = get_annotations(generator)
    else:
        all_detections = detections
        all_annotations = annotations

    return_list = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0), 3: (0, 0), 4: 0}
    average_precisions = {}
    conf_matrix = {}
    print('\nRecall and Precision', file=f)
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        label_name = generator.label_to_name(label)
        print(label_name)
        print("Recall: {}".format(max(recall, default=float('NaN'))), file=f)
        print("Precision: {}".format(min(precision, default=float('NaN'))), file=f)
        return_list[label] = (label_name, max(recall, default=float('NaN')), min(precision, default=float('NaN')))

        # compute average precision
        average_precision = _compute_ap(recall, precision, noise=noise, label=label, plot=plot)
        average_precisions[label] = average_precision, num_annotations

    print('\nAP:', file=f)
    map_list = []
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        map_list.append(average_precisions[label][0])
        print('{}: {}'.format(label_name, average_precisions[label][0]), file=f)
        return_list[2 + label] = (label_name, average_precisions[label][0])
    print('\nmAP: {}'.format(np.mean(map_list)), file=f)
    return_list[4] = np.mean(map_list)
    print('-----------------------------', file=f)
    return average_precisions, return_list
