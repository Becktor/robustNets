import argparse
import collections
import random

import torch.optim as optim
from torchvision import transforms
from network import retinanet, csv_eval, retinanet_normal
from network.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Crop, \
    crop_collater, LabelFlip
from torch.utils.data import DataLoader
from utils import *
import wandb
import time
import csv
import higher
from tqdm import tqdm
import plotly

assert torch.__version__.split('.')[0] == '1'


# if 'PYCHARM' in os.environ:
#    os.environ["WANDB_MODE"] = "dryrun"

def main(args=None):
    print(torch.__version__)
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--csv_weight', help='Path to file containing validation annotations')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=360)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--noise', help='Batch size', type=bool, default=False)
    parser.add_argument('--continue_training', help='Path to previous ckp', type=str, default=None)
    parser.add_argument('--base_model', help='Path base model ', type=str, default=None)
    parser.add_argument('--pre_trained', help='ResNet base pre-trained or not', type=bool, default=True)
    parser.add_argument('--label_flip', help='Label_flipping', type=bool, default=False)
    parser.add_argument('--flip_mod', help='dataloader flip modifier', type=int, default=0)
    parser.add_argument('--rew_start', help='reweight starting point', type=int, default=0)
    parser.add_argument('--reannotate', help='reannotate samples', type=bool, default=False)

    parser = parser.parse_args(args)

    reweight_mods = {0: "0%", 2: "50%", 3: "33%", 4: "25%"}

    reweight_mod = reweight_mods[parser.flip_mod]
    if parser.continue_training is not None:
        id = parser.continue_training[-8:]
        wandb.init(project="reanno", id=id, resume=True)
    else:
        wandb.init(project="reanno", config={
            "learning_rate": 5e-5,
            "ResNet": parser.depth,
            "reweight": parser.rew_start,
            "gamma": 0.1,
            "pre_trained": parser.pre_trained,
            "train_set": parser.csv_train,
            "batch_size": parser.batch_size,
            "reweight_mod": reweight_mod,
            "reanno": parser.reannotate
        })
    config = wandb.config
    wandb_name = wandb.run.name + "_" + wandb.run.id

    """
    Data loaders
    """

    trans = [LabelFlip(mod=parser.flip_mod), Crop(), Augmenter(), Resizer()]
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose(trans))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes)

    if parser.csv_weight is None:
        dataset_weight = None
        print('No weight annotations provided.')
    else:
        dataset_weight = CSVDataset(train_file=parser.csv_weight, class_list=parser.csv_classes,
                                    transform=transforms.Compose([Crop(reweight=True), Augmenter(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater,
                                  batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=True)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=crop_collater, batch_sampler=sampler_val)

    weighted_dataset_in_mem = {}

    if dataset_weight is not None:
        dataloader_weight = DataLoader(dataset_weight, batch_size=8,
                                       num_workers=4, collate_fn=collater)

        temp = []
        max_shape = -1
        for _ in tqdm(range(1)):
            for weighted_data in dataloader_weight:
                v_image, v_labels, w_names, idx, _ = weighted_data.as_batch()
                max_shape = v_labels.shape[1] if max_shape <= v_labels.shape[1] else max_shape
                temp.append((v_image, v_labels, w_names, idx))

        for weighted_data in temp:
            v_image, v_labels, w_names, idx = weighted_data
            for x in range(v_image.shape[0]):
                tmp = torch.ones((max_shape, 5)) * -1
                tmp[0:v_labels[x].shape[0], :] = v_labels[x]
                if idx[x] in weighted_dataset_in_mem:
                    weighted_dataset_in_mem[idx[x]].append((v_image[x], tmp.cuda(), w_names[x], idx[x]))
                else:
                    weighted_dataset_in_mem[idx[x]] = [(v_image[x], tmp.cuda(), w_names[x], idx[x])]

    pre_trained = False
    if parser.pre_trained:
        pre_trained = True
    # Create the model
    if parser.depth == 1:
        model = retinanet.rresnet18(num_classes=dataset_train.num_classes())
    elif parser.depth == 18:
        model = retinanet.resnet18(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 34:
        model = retinanet.resnet34(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 50:
        model = retinanet.resnet50(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 101:
        model = retinanet.resnet101(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 152:
        model = retinanet.resnet152(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    """
       Optimizer
    """
    checkpoint_dir = os.path.join('trained_models', wandb_name)

    # count_parameters(model)
    optimizer = optim.AdamW(model.params(), lr=config.learning_rate)

    # n_iters = len(dataset_train) / parser.batch_size
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size,
    #                                     gamma=config.gamma)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-5,
    #                                         step_size_up=n_iters, cycle_momentum=False)

    n_iters = int(len(dataset_train) / parser.batch_size)
    wr = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_iters * wr, 2)

    prev_epoch = 0
    if parser.continue_training is not None:
        model, optimizer, scheduler, checkpoint_dict = load_ckp(parser.continue_training, model, optimizer, scheduler)
        checkpoint_dir = parser.continue_training
        prev_epoch = checkpoint_dict['epoch']
        # mAP = checkpoint_dict['mAP']
    elif parser.base_model is not None:
        model = load_base_model(parser.base_model, model)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    else:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    wandb.watch(model)
    # scheduler.last_epoch = prev_epoch
    loss_hist = collections.deque(maxlen=500)

    model.training = True

    model.train()
    model.freeze_bn()

    print('Num training images: {} and num itr: {}'.format(len(dataset_train), n_iters))

    device = torch.device('cuda:0')

    zero_tensor = torch.tensor(0., device=device)
    mAP = 0
    use_gpu = True
    if use_gpu:
        model = model.cuda()
    model = model.cuda()
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(True)
    reweight_cases = {}
    altered_labels = {}
    dataset_len = len(dataloader_train)
    for epoch_num in range(parser.epochs):
        t0 = time.time()
        curr_epoch = prev_epoch + epoch_num
        model.train()
        # model.freeze_bn()
        epoch_loss = []
        m_epoch_loss = []
        print('============= Starting Epoch {} ============\n'.format(curr_epoch))
        skipped_iters = 0
        zero_loss = 0
        lr = get_lr(optimizer)

        if curr_epoch > 0:
            lr = get_lr(optimizer)
            print('setting LR: {}'.format(lr))
        for iter_num, data in enumerate(dataloader_train):
            image, labels, names, idxs, crop_ids = data.as_batch()

            # if curr_epoch == (wr-1):
            #     scheduler.base_lrs[0] = scheduler.base_lrs[0] * 0.5

            if curr_epoch >= config.reweight:
                if parser.reannotate:
                    for i, x in enumerate(idxs):
                        tst_key = "{}_{}".format(str(x), crop_ids[i])
                        if tst_key in altered_labels.keys():
                            alt = altered_labels[tst_key].shape
                            lbs = labels.data.shape
                            if alt[0] > lbs[1]:
                                temp = torch.ones([parser.batch_size, alt[0], alt[1]]) * -1
                                temp[:, 0:lbs[1], :] = labels.data
                                labels.data = temp.cuda()
                            elif alt[0] < lbs[1]:
                                labels.data[i][0:alt[0], :] = altered_labels[tst_key].cuda()
                            else:
                                labels.data[i] = altered_labels[tst_key].cuda()

            classification_loss, regression_loss, cl = model([image, labels])
            cost = cl[0] + cl[1]
            loss = torch.mean(cost)
            if loss == zero_tensor:
                zero_loss += 1
                continue

            if curr_epoch >= config.reweight:
                val_samples = get_random_weighting_sample(weighted_dataset_in_mem,
                                                          parser.batch_size)
                #rew_loss = reweight_loop(model, optimizer, image, labels, parser, val_samples, m_epoch_loss,
                #                         reweight_cases, names, trans, crop_ids, altered_labels, cost)
                rew_loss = reweight_loop_old(model, lr, image, labels, parser, val_samples, m_epoch_loss, zero_tensor,
                                             zero_loss, reweight_cases, names, trans, crop_ids, altered_labels, cost,
                                             dataset_train)
                if rew_loss:
                    loss = rew_loss
            # Lines 12 - 14 computing for the loss with the computed weights
            # and then perform a parameter update

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scheduler.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            runtime = (time.time() - t0) / (1 + iter_num)

            if iter_num % 2 == 0:
                lr = get_lr(optimizer)
                if len(m_epoch_loss) == 0:
                    m_epoch_loss.append(0)
                print(
                    'Itr: {} | Class loss: {:1.5f} | Reg loss: {:1.5f} | '
                    'mel: {:1.5f} el: {:1.5f} | LR: {} | rt : {:1.3f} est rem: {:1.3f} '.format(iter_num, float(
                        classification_loss),
                                                                                                float(regression_loss),
                                                                                                np.mean(m_epoch_loss),
                                                                                                np.mean(epoch_loss),
                                                                                                float(lr),
                                                                                                runtime, (runtime * (
                                dataset_len - iter_num))), end='\r')
            del classification_loss
            del regression_loss

        runtime = time.time() - t0
        print("\nEpoch {} took: {}".format(curr_epoch, runtime))

        # Reannotate poor annotations given reweight cases

        try:
            with open('reweight_cases.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in reweight_cases.items():
                    if value[0] > 1:
                        writer.writerow([key, value])
        except IOError:
            print("I/O error")

        if parser.csv_val is not None:
            print('Evaluating dataset')

            _ap, rl = csv_eval.evaluate(dataloader_val, model)

            # Write to Wandb
            wandb.log({"train/Epoch_runtime": runtime,
                       "train/running_loss": np.mean(loss_hist),
                       "train/epoch_loss": np.mean(epoch_loss),
                       "train/meta_epoch_loss": np.mean(m_epoch_loss),
                       "val/Buoy_Recall": rl[0][1],
                       "val/Buoy_Precision": rl[0][2],
                       "val/Boat_Recall": rl[1][1],
                       "val/Boat_Precision": rl[1][2],
                       "mAP/AP_Buoy": rl[0][3],
                       "mAP/AP_Boat": rl[1][3],
                       "mAP/mAP": rl['map'],
                       "mAP/mAP50": rl['map50'],
                       "lr/Learning Rate": lr,
                       'train/epoch': curr_epoch})

            checkpoint = {
                'epoch': curr_epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'buoy_AP': rl[0][3],
                'boat_AP': rl[1][3],
                'mAP': rl['map']
            }

            if rl['map'] > mAP:
                mAP = rl['map']
                save_ckp(checkpoint, model, True, checkpoint_dir, curr_epoch)
            else:
                save_ckp(checkpoint, model, False, checkpoint_dir, curr_epoch)

            loss_file = open(os.path.join(checkpoint_dir, "loss.csv"), "a+")
            loss_file.write("{}, {}, {}, {}, {}, {}\n".format(curr_epoch, np.mean(loss_hist),
                                                              rl[0], rl[1], rl['map'], rl['map50']))
            loss_file.close()
    model.eval()
    torch.save(model, 'model_final.pt')


def get_random_weighting_sample(weight_samples_dict, batch_size):
    val_samples = random.sample(weight_samples_dict.keys(), batch_size)
    tmp_samples = []
    v, l, n, i = [], [], [], []
    for x in val_samples:
        sample = random.sample(weight_samples_dict[x], 1)
        v.append(sample[0][0])
        l.append(sample[0][1])
        n.append(sample[0][2])
        i.append(sample[0][3])
    v = torch.stack(v)
    l = torch.stack(l)
    ret_samples = (v, l, n, i)
    return ret_samples


def reweight_loop(model, optimizer, image, labels, parser, val_sample, m_epoch_loss,
                  reweight_cases, names, trans, crop_ids, altered_labels, cost):
    with higher.innerloop_ctx(model, optimizer) as (meta_model, meta_opt):
        # y_f_hat = meta_net(image)
        _, _, meta_cl = meta_model([image, labels])
        meta_cost = meta_cl[0] + meta_cl[1]
        eps = torch.zeros(meta_cost.size()).cuda()
        eps = eps.requires_grad_()

        l_f_meta = torch.sum(meta_cost * eps)
        meta_model.zero_grad(set_to_none=True)
        meta_opt.step(l_f_meta)

        # reshuffle samples
        v_image, v_labels, w_names, idxs = val_sample
        g_meta_classification_loss, g_meta_regression_loss, _ = meta_model([v_image, v_labels])
        l_g_meta = g_meta_classification_loss + g_meta_regression_loss
        m_epoch_loss.append(float(l_g_meta))
        grad_eps = torch.autograd.grad(l_g_meta, eps)[0].detach()
        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)
        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        wl = torch.le(w, 0.5/eps.shape[0])
        ####
        ### meta annotation
        ####
        update_anno = np.full(parser.batch_size, False)

        add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, names, trans, idxs, update_anno)

        meta_model.eval()
        if update_anno.sum() > 0:
            update_annotation(image, update_anno, idxs, crop_ids, meta_model, altered_labels)
        loss = torch.sum(cost * w)
    return loss


def reweight_loop_old(model, lr, image, labels, parser, val_sample, m_epoch_loss, zero_tensor,
                      zero_loss, reweight_cases, names, trans, crop_ids, altered_labels, cost, dataset_train):
    meta_model = retinanet.resnet50(num_classes=dataset_train.num_classes())
    meta_model.load_state_dict(model.state_dict())
    if torch.cuda.is_available():
        meta_model.cuda()
    _, _, meta_cl = meta_model([image, labels])
    meta_cost = meta_cl[0] + meta_cl[1]
    eps = torch.zeros(meta_cost.size()).cuda()
    eps = eps.requires_grad_()

    l_f_meta = torch.sum(meta_cost * eps)
    meta_model.zero_grad(set_to_none=True)

    # Get original gradients with epsilon applied.
    grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True, allow_unused=True)
    if any(x is None for x in grads):
        return

    meta_model.update_params(lr, source_params=grads)
    # reshuffle samples
    v_image, v_labels, w_names, idxs = val_sample
    g_meta_classification_loss, g_meta_regression_loss, _ = meta_model([v_image, v_labels])
    l_g_meta = g_meta_classification_loss + g_meta_regression_loss
    m_epoch_loss.append(float(l_g_meta))
    grad_eps = torch.autograd.grad(l_g_meta, eps)[0].detach()
    # Line 11 computing and normalizing the weights
    w_tilde = torch.clamp(-grad_eps, min=0)
    norm_c = torch.sum(w_tilde) + 1e-10
    if norm_c != 0:
        w = w_tilde / norm_c
    else:
        w = w_tilde

    wl = torch.le(w, 0.5/eps.shape[0])
    ### meta annotation

    update_anno = np.full(parser.batch_size, False)

    add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, names, trans, idxs, update_anno)

    meta_model.eval()
    if update_anno.sum() > 0:
        update_annotation(image, update_anno, idxs, crop_ids, meta_model, altered_labels)
    loss = torch.sum(cost * w)

    if loss == zero_tensor:
        zero_loss += 1
    return loss


def add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, names, trans, idxs, update_anno):
    for index, weight in enumerate(w):
        if wl[index]:
            if names[index] in reweight_cases:
                tmp_cnt, tmp_loss = reweight_cases[names[index]]
                tmp_loss.append(float(weight))
                sample = (tmp_cnt + 1, tmp_loss)
                reweight_cases[names[index]] = sample
                if tmp_cnt > 0:  # arbitrarily chosen
                    trans[0].alt(idxs[index], sample)
                    update_anno[index] = True
            else:
                reweight_cases[names[index]] = (1, [float(weight)])
                # trans[0].alt(idxs[index], sample)
                update_anno[index] = True


def update_annotation(image, update_anno, idxs, crop_ids, meta_model, altered_labels):
    update_img = image[update_anno]
    update_names = np.array(idxs)[update_anno]
    update_crop_ids = np.array(crop_ids)[update_anno]
    score, classes, bbox = meta_model(update_img)
    for i in range(len(score)):
        if score[i][0] != -1:
            c = classes[i]
            b = bbox[i]
            new_anno = torch.cat([b, c.reshape([-1, 1])], axis=1)
            key = "{}_{}".format(update_names[i], update_crop_ids[i])
            altered_labels[key] = new_anno.detach()


def update_params(model, lr_inner, source_params=None):
    named_params = model.named_parameters()
    if source_params is not None:
        for tgt, src in zip(named_params, source_params):
            name_t, param_t = tgt
            grad = src
            tmp = param_t - lr_inner * grad
            vtmp = to_var(tmp)
            param_t.data = vtmp


if __name__ == '__main__':
    main()
