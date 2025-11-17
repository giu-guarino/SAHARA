ds = ["EUROSAT-MS-SAR", "RESISC45_EURO", "SUNRGBD", "TRISTAR", "HANDS", "AV-MNIST", "CREMA-D"]  #datasets

data_names = {
    "EUROSAT-MS-SAR": ["MS", "SAR"],
    "RESISC45_EURO": ["RESISC45", "EURO"], #RGB - MS
}

backbones_list = ["CNN", "ViT"]

TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LEARNING_RATE_DC = 1e-3
MOMENTUM_EMA = .95
EPOCHS =  200
WARM_UP_EPOCH_EMA = 50

GP_PARAM = 10
DC_PARAM = 0.1
ITER_DC = 10 # Inner iterations for domain critic module

TH_FIXMATCH = .95

decouple_ds = True

#TinyViT 5M parameters

TV_param = {
    "embed_dims": [64, 128, 160, 320],
    "depths": [2, 2, 6, 2],
    "num_heads": [2, 4, 5, 10],
    "window_sizes": [7, 7, 14, 7],
    "drop_path_rate": 0.0,
}


'''
#TinyViT 11M parameters
TV_param = {
    "embed_dims": [64, 128, 256, 448],
    "depths": [2, 2, 6, 2],
    "num_heads": [2, 4, 8, 14],
    "window_sizes": [7, 7, 14, 7],
    "drop_path_rate": 0.1,
}
'''
#embed_dims=[32, 64, 96, 160]
#num_heads=[1, 2, 3, 5]

