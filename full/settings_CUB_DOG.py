img_size = 224
prototype_shape = (1000*1, 64, 1, 1)
# prototype_shape = (1000*4, 64, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path = '/workspace/ST-ProtoPNet/data/CUB_200_2011/dataset/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test_full/'
train_push_dir = data_path + 'train'

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 100

joint_optimizer_lrs = {'features': 1 * 1e-4,
                       'hyper_params': 1 * 5e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

joint_lr_step_size = 10

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'hyper_params': 1 * 5e-4,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst_trv': -0.8,
    'sep_trv': 0.08,
    'clst_spt': -0.8,
    'sep_spt': 0.48,
    'orth': 1e-3,
    'l1': 1e-4,
    'discr': 1,
    'close': 1,
    'entailment': 0.2,
}


num_train_epochs = 120
num_warm_epochs = 10

push_start = 121
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]     # 10

