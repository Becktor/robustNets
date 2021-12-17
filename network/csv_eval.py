from __future__ import print_function

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import wandb
from network.dataloader import Sample


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

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(
        np.expand_dims(a[:, 0], 1), b[:, 0]
    )
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(
        np.expand_dims(a[:, 1], 1), b[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = (
        np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1)
        + area
        - iw * ih
    )

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    save_dir=".",
    names=(),
    wdb=True,
    name="",
):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    tp = np.expand_dims(tp, axis=1)
    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [
        v for k, v in names.items() if k in unique_classes
    ]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot and len(py) != 0:
        pr_p = plot_pr_curve(px, py, ap, "PR_curve.png", names, wdb=wdb, name=name)
        f1_p = plot_mc_curve(
            px, f1, "F1_curve.png", names, ylabel="F1", wdb=wdb, name=name
        )
        p_p = plot_mc_curve(
            px, p, "P_curve.png", names, ylabel="Precision", wdb=wdb, name=name
        )
        r_p = plot_mc_curve(
            px, r, "R_curve.png", names, ylabel="Recall", wdb=wdb, name=name
        )

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=(), wdb=True, name=""):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label="all classes %.3f mAP@0.5" % ap[:, 0].mean(),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.7, 1), loc="upper left")
    if wdb:
        f_img = wandb.Image(fig)
        wandb.log({save_dir: f_img})
    else:
        fig.savefig(name + "_" + save_dir, dpi=250)
    plt.close()
    return fig


def plot_mc_curve(
    px,
    py,
    save_dir="mc_curve.png",
    names=(),
    xlabel="Confidence",
    ylabel="Metric",
    wdb=True,
    name="",
):
    # Metric-confidence curve4
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.7, 1), loc="upper left")
    if wdb:
        f_img = wandb.Image(fig)
        wandb.log({save_dir: f_img})
    else:
        fig.savefig(name + "_" + save_dir, dpi=250)
    plt.close()
    return fig


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def _compute_ap(recall, precision, noise=0, label="", plot=False):
    """Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    if plot:
        # Plot pres vs recall
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        label_str = "Buoy" if label == 0 else "Boat"
        if noise <= 0.0:
            ax.set_title(
                "Precision-Recall curve of {}".format(label_str, noise),
                loc="center",
                y=1.07,
            )
        else:
            ax.set_title(
                "Precision-Recall curve of {} with noise {}".format(label_str, noise),
                loc="center",
                y=1.07,
            )

        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        minval = np.min(mpre[np.nonzero(mpre)])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black", alpha=0.5)
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.5)
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


def get_detections(
    dataloader, model, score_threshold=0.05, max_detections=100, noise_level=0.0
):
    """Get the detections from the network using the generator.
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
    dataset = []
    for index, data in enumerate(dataloader):
        for elem in data:
            dataset.append(elem)
    ds = dataloader.dataset
    all_detections = [
        [None for i in range(ds.num_classes())] for j in range(len(dataset))
    ]
    all_annotations = [
        [None for i in range(ds.num_classes())] for j in range(len(dataset))
    ]
    model.eval()

    with torch.no_grad():
        for index, data in tqdm(enumerate(dataset)):
            annotation = data.annot.numpy()
            img = data.img

            for label in range(ds.num_classes()):
                all_annotations[index][label] = annotation[
                    annotation[:, 4] == label, :4
                ].copy()
            # if noise_level > 0:
            #   img = img + torch.empty(img.shape).normal_(mean=0, std=noise_level).numpy()

            # run network
            try:
                scores, labels, boxes = model(img.cuda().float().unsqueeze(dim=0))
                if len(scores) <= 0:
                    scores = scores.cpu().numpy()
                    labels = labels.cpu().numpy()
                    boxes = boxes.cpu().numpy()
                else:
                    scores = scores[0].cpu().numpy()
                    labels = labels[0].cpu().numpy()
                    boxes = boxes[0].cpu().numpy()
            except Exception as e:
                print(e)
            # correct boxes for image scale
            # boxes /= scale
            debug = False

            if debug:
                import matplotlib.pyplot as plt
                import cv2

                img2 = img.numpy()
                bs = zip(boxes, scores)
                for v, s in bs:
                    if s > 0.3:
                        img2 = cv2.rectangle(
                            img2,
                            (int(v[0]), int(v[1])),
                            (int(v[2]), int(v[3])),
                            color=(0, 0, 1),
                            thickness=2,
                        )
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
                    [
                        image_boxes,
                        np.expand_dims(image_scores, axis=1),
                        np.expand_dims(image_labels, axis=1),
                    ],
                    axis=1,
                )

                # copy detections to all_detections
                for label in range(ds.num_classes()):
                    all_detections[index][label] = image_detections[
                        image_detections[:, -1] == label, :-1
                    ]
            else:
                # copy detections to all_detections
                for label in range(ds.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            # print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections, all_annotations


def get_annotations(generator):
    """Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [
        [None for i in range(generator.num_classes())] for j in range(len(generator))
    ]

    for i in range(len(generator)):
        # load the annotations

        img = generator.load_image(i)
        annotations = generator.load_annotations(i, img.shape)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[
                annotations[:, 4] == label, :4
            ].copy()

        print("{}/{}".format(i + 1, len(generator)), end="\r")

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.001,
    max_detections=100,
    noise=0,
    save_path=None,
    detections=None,
    annotations=None,
    f=None,
    plot=False,
    wdb=True,
    name="",
):
    """Evaluate a given dataset using a given network.
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
        all_detections, all_annotations = get_detections(
            generator,
            model,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
    else:
        all_detections = detections
        all_annotations = annotations

    return_list = {0: (0, 0, 0), 1: (0, 0, 0), "map": 0, "map50": "0"}
    average_precisions = {}
    all_pred_lbl = []
    all_actual_labels, all_scores, all_true_positives = [], [], []

    print("\nRecall and Precision", file=f)
    for label in range(generator.dataset.num_classes()):
        false_positives = []  # np.zeros((len(all_detections),))
        true_positives = []  # np.zeros((len(all_detections),))
        scores = []
        pred_labels = []
        actual_labels = []
        num_annotations = 0.0
        for i in range(len(all_detections)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            for _ in range(annotations.shape[0]):
                actual_labels.append(label)
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                pred_labels.append(label)
                if annotations.shape[0] == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if (
                    max_overlap >= iou_threshold
                    and assigned_annotation not in detected_annotations
                ):
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        all_actual_labels.append(actual_labels)
        all_pred_lbl.append(pred_labels)
        all_scores.append(scores)
        all_true_positives.append(true_positives)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

    names = {}
    for label in range(generator.dataset.num_classes()):
        label_name = generator.dataset.label_to_name(label)
        names[label] = label_name

    p, r, ap, f1, ap_class = ap_per_class(
        np.concatenate(all_true_positives),
        np.concatenate(all_scores),
        np.concatenate(all_pred_lbl),
        np.concatenate(all_actual_labels),
        plot=True,
        names=names,
        wdb=wdb,
        name=name,
    )
    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    for label in range(generator.dataset.num_classes()):
        label_name = generator.dataset.label_to_name(label)
        print("AP-{}: {}".format(label_name, ap[label]), file=f)
        print("recall-{}: {}".format(label_name, r[label]), file=f)
        print("precision-{}: {}".format(label_name, p[label]), file=f)
        return_list[label] = (label_name, r[label], p[label], ap[label])
    print("\nmAP50: {}, mAP: {}".format(map50, map), file=f)
    return_list["map"] = map
    return_list["map50"] = map50
    print("-----------------------------", file=f)

    return ap, return_list
