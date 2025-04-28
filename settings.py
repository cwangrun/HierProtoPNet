base_architecture = 'fpndensenet121'
img_size = 512
num_classes = 15
prototype_shape = (50*num_classes, 256, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = str(img_size)

root_dir = '/mnt/c/chong/data/chestxray/'

train_batch_size = 32        # 64, 32
test_batch_size = 32         # 200
train_push_batch_size = 32   # 100


lr_backbone = 1e-4
lr = 3e-3

joint_optimizer_lrs = {'features': lr_backbone,
                       'add_on_layers': lr,
                       'prototype_vectors': lr}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': lr,
                      'prototype_vectors': lr}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1.0,
    'clst': 0.02,
    'sep': 0.02,
    'mining': 0.1,
    'distil': 0.1
}



