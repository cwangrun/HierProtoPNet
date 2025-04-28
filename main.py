import os
import torch
import time
import argparse
import re
import wandb
from helpers import makedir
import model
import push_high
import push_mid
import push_low
import train_and_test as tnt
import save
from log import create_logger
from utlis.utlis_func import *


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')     # "0, 1"
parser.add_argument('-seed', type=int, default=42)
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])

os.environ['WANDB_START_METHOD'] = 'fork'
os.environ["WANDB__SERVICE_WAIT"] = "1500"

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, coefs, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


model_dir = 'saved_models/{}/'.format(datestr()) + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import root_dir, train_batch_size, test_batch_size, train_push_batch_size

args.train_batch_size = train_batch_size
args.test_batch_size = test_batch_size
args.train_push_batch_size = train_push_batch_size
args.coefs = coefs
args.num_classes = num_classes
args.img_size = img_size
args.root_dir = root_dir
args.model_dir = model_dir


train_loader, train_push_loader, test_loader, valid_loader = config_dataset(args)


# WandB – Initialize a new run
wandb.init(project='HierProtoPNet-Xray', mode='disabled')     # mode='disabled'
wandb.run.name = wandb.run.id + '_low'

# construct the model
ppnet = model.build_HierProtoPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True


# define optimizer
from settings import warm_optimizer_lrs
from settings import joint_optimizer_lrs, joint_lr_step_size
from settings import last_layer_optimizer_lr


weight_decay = 0e-3

# train the model
###################################################################################################################
log('start training high level')
joint_optimizer_specs_high = \
[
 {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': weight_decay}, # bias are now also being regularized
 {'params': ppnet.add_on_layers_high.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay},
 {'params': ppnet.prototype_vectors_high, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer_high = torch.optim.Adam(joint_optimizer_specs_high)
joint_lr_scheduler_high = torch.optim.lr_scheduler.StepLR(joint_optimizer_high, step_size=joint_lr_step_size, gamma=0.99)   # 0.1

warm_optimizer_specs_high = \
[{'params': ppnet.add_on_layers_high.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay},
 {'params': ppnet.prototype_vectors_high, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer_high = torch.optim.Adam(warm_optimizer_specs_high)

last_layer_optimizer_specs_high = \
[
 {'params': ppnet.last_layer_high.parameters(), 'lr': last_layer_optimizer_lr},
]
last_layer_optimizer_high = torch.optim.Adam(last_layer_optimizer_specs_high)
num_warm_epochs_high = 2   # 5
num_train_epochs_high = 35
push_start_high = 30  # 10, 15, 80
push_epochs_high = [i for i in range(num_train_epochs_high) if i % 5 == 0]
for epoch in range(num_train_epochs_high):
    log('epoch of high: \t{0}'.format(epoch))

    if epoch < num_warm_epochs_high:
        tnt.warm_only_high(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer_high, train_scale='high',
                      coefs=coefs, log=log)
    else:
        tnt.joint_high(model=ppnet_multi, log=log)
        joint_lr_scheduler_high.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer_high, train_scale='high',
                      coefs=coefs, log=log)
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'high_nopush', accu=accu,
                                target_accu=0.70, log=log)

    if epoch >= push_start_high and epoch in push_epochs_high:
        push_high.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=None,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + '/high',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'high_push', accu=accu,
                                    target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only_high(model=ppnet_multi, log=log)
            for i in range(6):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer_high,
                              train_scale='high', train_last=True, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'high-push', accu=accu,
                                            target_accu=0.70, log=log)
################################################################################################################
log('start training middle level')
joint_optimizer_specs_middle = \
    [
        {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': weight_decay},  # bias are now also being regularized
        {'params': ppnet.add_on_layers_middle.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay},
        {'params': ppnet.prototype_vectors_middle, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
joint_optimizer_middle = torch.optim.Adam(joint_optimizer_specs_middle)
joint_lr_scheduler_middle = torch.optim.lr_scheduler.StepLR(joint_optimizer_middle, step_size=1, gamma=0.5)  # 0.1, 0.99

last_layer_optimizer_specs_middle = \
    [
        {'params': ppnet.last_layer_middle.parameters(), 'lr': last_layer_optimizer_lr},
    ]
last_layer_optimizer_middle = torch.optim.Adam(last_layer_optimizer_specs_middle)
num_train_epochs_middle = 20
push_start_middle = 15  # 10, 15, 80
push_epochs_middle = [i for i in range(num_train_epochs_middle) if i % 5 == 0]
for epoch in range(num_train_epochs_middle):
    log('epoch of middle: \t{0}'.format(epoch))

    tnt.joint_middle(model=ppnet_multi, log=log)
    log('##### lr: \t{0}'.format(joint_optimizer_middle.param_groups[0]['lr']))
    if epoch in [9, 12, 15]:
        joint_lr_scheduler_middle.step()
    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer_middle, train_scale='middle',
                  coefs=coefs, epoch=epoch, log=log)
    wandb.log({
        "LR": joint_optimizer_middle.param_groups[0]['lr'],
        "Epoch": epoch,
    })
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'middle_nopush', accu=accu,
                                target_accu=0.70, log=log)

    if epoch >= push_start_middle and epoch in push_epochs_middle:
        push_mid.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=None,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + '/middle',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'middle_push', accu=accu,
                                    target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only_middle(model=ppnet_multi, log=log)
            for i in range(6):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer_middle,
                              train_scale='middle', train_last=True, coefs=coefs,
                              epoch=epoch, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'middle-push', accu=accu,
                                            target_accu=0.70, log=log)
####################################################################################################################
log('start training low level')
joint_optimizer_specs_low = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': weight_decay}, # bias are now also being regularized
 {'params': ppnet.add_on_layers_low.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay},
 {'params': ppnet.prototype_vectors_low, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer_low = torch.optim.Adam(joint_optimizer_specs_low)
joint_lr_scheduler_low = torch.optim.lr_scheduler.StepLR(joint_optimizer_low, step_size=1, gamma=0.5)   # 0.1, 0.99

last_layer_optimizer_specs_low = \
[
 {'params': ppnet.last_layer_low.parameters(), 'lr': last_layer_optimizer_lr}
]
last_layer_optimizer_low = torch.optim.Adam(last_layer_optimizer_specs_low)
num_train_epochs_low = 20
push_start_low = 15  # 10, 15, 80
push_epochs_low = [i for i in range(num_train_epochs_low) if i % 5 == 0]
for epoch in range(num_train_epochs_low):
    log('epoch of low: \t{0}'.format(epoch))

    tnt.joint_low(model=ppnet_multi, log=log)
    if epoch in [9, 12, 15]:
        joint_lr_scheduler_low.step()
    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer_low, train_scale='low',
                  coefs=coefs, epoch=epoch, log=log)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(16) + 'low_nopush', accu=0.86,
                                target_accu=0.70, log=log)

    if epoch >= push_start_low and epoch in push_epochs_low:
        push_low.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=None,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir + '/low',  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(16) + 'low_push', accu=0.86,
                                    target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only_low(model=ppnet_multi, log=log)
            for i in range(6):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer_low,
                              train_scale='low', train_last=True, coefs=coefs, epoch=epoch, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'low-push', accu=accu,
                                            target_accu=0.70, log=log)

logclose()

