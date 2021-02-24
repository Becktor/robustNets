import argparse
import collections
import datetime as dt
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from network import retinanet, csv_eval
from network.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Crop, \
    crop_collater_for_validation, LabelFlip
from torch.utils.data import DataLoader
from utils import *
import wandb
import time
import copy

assert torch.__version__.split('.')[0] == '1'


# if 'PYCHARM' in os.environ:
#     os.environ["WANDB_MODE"] = "dryrun"

def main(args=None):
    print(torch.__version__)
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--csv_weight', help='Path to file containing validation annotations')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=500)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=10)
    parser.add_argument('--noise', help='Batch size', type=bool, default=False)
    parser.add_argument('--continue_training', help='Path to previous ckp', type=str, default=None)
    parser.add_argument('--pre_trained', help='ResNet base pre-trained or not', type=bool, default=True)
    parser.add_argument('--label_flip', help='ResNet base pre-trained or not', type=bool, default=True)

    parser = parser.parse_args(args)

    if parser.continue_training is not None:
        id = parser.continue_training[-8:]
        wandb.init(project="reweight", id=id, resume=True)
    else:
        wandb.init(project="reweight", config={
            "learning_rate": 1e-4,
            "ResNet": parser.depth,
            "reweight": 0,
            "milestones": [10, 75, 100],
            "gamma": 0.1,
            "pre_trained": parser.pre_trained,
            "train_set": parser.csv_train,
            "batch_size": parser.batch_size,
            "label_flip": parser.label_flip
        })
    config = wandb.config
    wandb_name = wandb.run.name + "_" + wandb.run.id
    """
    Data loaders
    """
    if parser.label_flip:
        trans = [LabelFlip(), Crop(), Augmenter(), Resizer()]
    else:
        trans = [Crop(), Augmenter(), Resizer()]
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, use_path=True,
                               transform=transforms.Compose(trans))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes)

    if parser.csv_weight is None:
        dataset_weight = None
        print('No validation annotations provided.')
    else:
        dataset_weight = CSVDataset(train_file=parser.csv_weight, class_list=parser.csv_classes, use_path=True,
                                    transform=transforms.Compose([Crop(), Augmenter(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater,
                                  batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=True)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=crop_collater_for_validation,
                                    batch_sampler=sampler_val)

    dataloader_weight = DataLoader(dataset_weight, batch_size=parser.batch_size, num_workers=3, collate_fn=collater,
                                   shuffle=True)

    pre_trained = False
    if parser.pre_trained:
        pre_trained = True
    # Create the model
    if parser.depth == 1:
        model = retinanet.rresnet18(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
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

    count_parameters(model)
    optimizer = optim.AdamW(model.params(), lr=config.learning_rate)

    n_iters = len(dataset_train) / parser.batch_size
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-5,
                                            step_size_up=n_iters, cycle_momentum=False)

    prev_epoch = 0
    if parser.continue_training is None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    else:
        model, optimizer, scheduler, checkpoint_dict = load_ckp(parser.continue_training, model, optimizer, scheduler)
        checkpoint_dir = parser.continue_training
        prev_epoch = checkpoint_dict['epoch']

        # mAP = checkpoint_dict['mAP']
    wandb.watch(model)
    # scheduler.last_epoch = prev_epoch
    loss_hist = collections.deque(maxlen=500)

    model.training = True

    model.train()
    model.freeze_bn()

    print('Num training images: {} and num itr: {}'.format(len(dataset_train), n_iters))

    zero_tensor = torch.tensor(0., device=torch.device('cuda'))
    mAP = 0
    use_gpu = True
    if use_gpu:
        model = model.cuda()
    model = model.cuda()
    torch.backends.cudnn.benchmark = True
    for epoch_num in range(parser.epochs):
        t0 = time.time()
        curr_epoch = prev_epoch + epoch_num
        model.train()
        model.freeze_bn()
        epoch_loss = []
        print('============= Starting Epoch {} ============\n'.format(curr_epoch))
        skipped_iters = 0
        zero_loss = 0
        lr = get_lr(optimizer)

        if curr_epoch > 0:
            lr = get_lr(optimizer)
            print('setting LR: {}'.format(lr))
        for iter_num, data in enumerate(dataloader_train):
            image = to_var(data['img'], requires_grad=False)
            labels = to_var(data['annot'], requires_grad=False)
            # names = data['name']
            classification_loss, regression_loss, cl = model([image, labels])
            cost = cl[0] + cl[1]
            loss = torch.mean(cost)
            if loss == zero_tensor:
                zero_loss += 1
                continue
            if curr_epoch >= config.reweight:
                # Line 2 get batch of data
                # initialize a dummy network for the meta learning of the weights
                # Setup meta net
                meta_model = copy.deepcopy(model)
                meta_model.cuda()

                # Lines 4 - 5 initial forward pass to compute the initial weighted loss

                meta_classification_loss, meta_regression_loss, meta_cl = meta_model([image, labels])
                meta_joined_cost = meta_cl[0] + meta_cl[1]
                # Get loss and apply epsilon.
                eps = to_var(torch.zeros(parser.batch_size))
                l_f_meta = torch.sum(meta_joined_cost * eps)
                meta_model.zero_grad(set_to_none=True)

                # Get original gradients with epsilon applied.
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True, allow_unused=True)
                if any(x is None for x in grads):
                    skipped_iters += 1
                    continue
                # Perform a parameter update
                meta_model.update_params(lr, source_params=grads)

                for weighted_data in dataloader_weight:
                    # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
                    v_image = to_var(weighted_data['img'], requires_grad=False)
                    v_labels = to_var(weighted_data['annot'], requires_grad=False)
                    names = weighted_data['name']
                    y_meta_classification_loss, y_meta_regression_loss, _ = meta_model([v_image, v_labels])
                    l_g_meta = y_meta_classification_loss + y_meta_regression_loss

                    grad_eps = torch.autograd.grad(l_g_meta.mean(), eps, only_inputs=True)[
                        0]  # find gradients with regard to epsilon

                    # Line 11 computing and normalizing the weights
                    w_tilde = torch.clamp(-grad_eps.detach(), min=0)
                    norm_c = torch.sum(w_tilde)

                    if norm_c != 0:
                        w = w_tilde / norm_c
                    else:
                        w = w_tilde
                    loss = torch.sum(cost * w)
                    break

                if loss == zero_tensor:
                    zero_loss += 1

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
                print(
                    'Itr: {} | Class loss: {:1.5f} | Reg loss: {:1.5f} | '
                    'rl: {:1.5f} | LR: {} | rt : {:1.3f} '.format(iter_num, float(classification_loss),
                                                                  float(regression_loss), np.mean(loss_hist), float(lr),
                                                                  runtime), end='\r')
            del classification_loss
            del regression_loss
        runtime = time.time() - t0
        print("\nEpoch {} took: {}".format(curr_epoch, runtime))
        if parser.csv_val is not None:
            print('Evaluating dataset')

            _ap, rl = csv_eval.evaluate(dataloader_val, model, 0.3, 0.3)

            # Write to Tensorboard
            wandb.log({"train/Epoch_runtime": runtime})
            wandb.log({"train/running_loss": np.mean(loss_hist)})

            wandb.log({"val/Buoy_Recall": rl[0][1]})
            wandb.log({"val/Buoy_Precision": rl[0][2]})

            wandb.log({"val/Boat_Recall": rl[1][1]})
            wandb.log({"val/Boat_Precision": rl[1][2]})

            wandb.log({"mAP/AP_Buoy": rl[2][1]})
            wandb.log({"mAP/AP_Boat": rl[3][1]})
            wandb.log({"mAP/mAP": rl[4]})

            wandb.log({"lr/Learning Rate": lr})

            checkpoint = {
                'epoch': curr_epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'buoy_AP': rl[2][1],
                'boat_AP': rl[3][1],
                'mAP': rl[4]
            }

            if rl[4] > mAP:
                mAP = rl[4]
                save_ckp(checkpoint, model, True, checkpoint_dir, curr_epoch)
            else:
                save_ckp(checkpoint, model, False, checkpoint_dir, curr_epoch)

            loss_file = open(os.path.join(checkpoint_dir, "loss.csv"), "a+")
            loss_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(curr_epoch, np.mean(loss_hist),
                                                                          rl[0], rl[1], rl[2], rl[3],
                                                                          rl[2][1], rl[3][1], rl[4]))
            loss_file.close()
    model.eval()
    torch.save(model, 'model_final.pt')


if __name__ == '__main__':
    main()
