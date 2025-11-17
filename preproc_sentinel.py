import numpy
import torch
import geopandas
from interpolator_tools import ideal_interpolator, mtf
import os
import numpy as np
from torch.nn import functional as func

'''
1. Latitude
2. Longitude
3. Type
4. Source
5. Image_Folder
6. geometry
'''

#file = open("/home/giuseppe/Datasets/train_dataset.geojson")
#df = geopandas.read_file(file)
#fields = df.columns.tolist()

def z_normalization(image):

    #mean = image.mean(axis=(0, 1, 2, 3), keepdims=True)  # Calcola la media per ogni canale
    #std = image.std(axis=(0, 1, 2, 3), keepdims=True)  # Calcola la deviazione standard per ogni canale

    mean = image.mean()  # Un solo valore scalare per tutto il dataset
    std = image.std()

    if std == 0:
        std = 1  # Evita divisioni per zero

    z_norm = (image - mean) / std  # Z-Normalization

    return (z_norm - np.min(z_norm)) / (np.max(z_norm) - np.min(z_norm))


def normalize_sentinel1(s1):
    """
    Normalizza i dati Sentinel-1 tra 0 e 1.

    :param image: array numpy dei dati Sentinel-1 in dB.
    :param band_type: 'VV' o 'VH', usa range differenti.
    :return: immagine normalizzata tra 0 e 1.
    """

    vv = s1[:, 0, None, :, :]
    vh = s1[:, 1, None, :, :]

    # Imposta il range di normalizzazione in base alla polarizzazione
    min_db1, max_db1 = (-25, 0)
    min_db2, max_db2 = (-32, -5)

    # Clipping per evitare outliers
    vv = np.clip(vv, min_db1, max_db1)
    vh = np.clip(vh, min_db2, max_db2)

    # Normalizzazione lineare
    vv = (vv - min_db1) / (max_db1 - min_db1)
    vh = (vh - min_db2) / (max_db2 - min_db2)

    return np.concatenate((vv, vh), axis=1)


def load_geojson(file_path):
    """Carica un file GeoJSON come GeoDataFrame."""
    try:
        file = open(file_path)
        gdf = geopandas.read_file(file)
        print(f"File '{file_path}' correctly loaded!")
        return gdf
    except Exception as e:
        print(f"Error during loading: {e}")
        return None


def process_images_and_labels(root, gdf):
    """Apre le immagini e crea gli array di immagini e labels."""
    s1_list = []
    s2_list = []
    label_list = []
    label_names = []

    # Definizione delle classi target
    classes = ["Negative", "CAFOs", "WWTreatment", "Landfills", "RefineriesAndTerminals", "ProcessingPlants", "Mines"]
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}  # Mappatura classe → indice

    for raw_idx, row in gdf.iterrows():

        label = row["Type"]
        img_path = row["Image_Folder"]

        print(img_path)

        # Considera solo le immagini con label nel set specificato
        if label in class_to_index and os.path.exists(os.path.join(root,img_path)) and img_path != "train_images/38.8205231_-75.7927729":

            print(f"Processing Image: {raw_idx}")

            s2_10 = np.load(os.path.join(root, img_path, "sentinel-2-10m.npy"))  # Legge l'immagine
            s2_20 = np.load(os.path.join(root, img_path, "sentinel-2-20m.npy"))  # Legge l'immagine
            #s2_60 = np.load(os.path.join(root, img_path, "sentinel-2-60m.npy"))  # Legge l'immagine
            s1 = np.load(os.path.join(root, img_path, "sentinel-1.npy"))  # Legge l'immagine

            s2_10 = np.moveaxis(s2_10, -1, 0)[None, :, :, :].astype(np.float32)
            s2_20 = np.moveaxis(s2_20, -1, 0)[None, :, :, :].astype(np.float32)
            #s2_60 = np.moveaxis(s2_60, -1, 0)[None, :, :, :].astype(np.float32)

            s1 = np.moveaxis(s1, -1, 0)[None, :, :, :].astype(np.float32)

            s2_10 = torch.from_numpy(s2_10).float()
            #s2_60 = torch.from_numpy(s2_60).float()

            s2_10_lp = mtf(s2_10, 'S2_10', 2, mode='edge')
            s2_10_lr = func.interpolate(s2_10_lp, scale_factor=1/2, mode='nearest-exact')
            #s2_60_exp = func.interpolate(s2_60, scale_factor=3, mode='bicubic')

            s2_10_lr = s2_10_lr.numpy()
            #s2_60_exp = s2_60_exp.numpy()

            '''
            s2 = np.concatenate((s2_60_exp[:, 0, None, :, :],
                                 s2_10_lr[:, 0, None, :, :], s2_10_lr[:, 1, None, :, :], s2_10_lr[:, 2, None, :, :],
                                 s2_20[:, 0, None, :, :], s2_20[:, 1, None, :, :], s2_20[:, 2, None, :, :],
                                 s2_10_lr[:, 3, None, :, :], s2_20[:, 3, None, :, :], s2_60_exp[:, 1, None, :, :],
                                 s2_60_exp[:, 2, None, :, :], s2_20[:, 4, None, :, :], s2_20[:, 5, None, :, :]),
                                 axis=1)            
            '''
            s2 = np.concatenate((s2_10_lr[:, 0, None, :, :], s2_10_lr[:, 1, None, :, :], s2_10_lr[:, 2, None, :, :],
                                 s2_20[:, 0, None, :, :], s2_20[:, 1, None, :, :], s2_20[:, 2, None, :, :],
                                 s2_10_lr[:, 3, None, :, :], s2_20[:, 3, None, :, :],
                                 s2_20[:, 4, None, :, :], s2_20[:, 5, None, :, :]),
                                 axis=1)

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converti da BGR a RGB
            #s1 = normalize_sentinel1(s1.astype(np.float32))
            #s2 = s2.astype(np.float32) / 2**12  # Normalizzazione tra 0 e 1

            s1_list.append(s1)
            s2_list.append(s2)
            label_list.append(class_to_index[label])

    s1_np = np.array(np.concatenate(s1_list, axis=0))  # [N, H, W, C]
    s2_np = np.array(np.concatenate(s2_list, axis=0))  # [N, H, W, C]
    labels_array = np.array(label_list, dtype=np.int64)

    return s1_np, s2_np, labels_array,


if __name__ == "__main__":

    root = "/home/giuseppe/Datasets/METER-ML_raw/"


    file_path_tr = "/home/giuseppe/Datasets/METER-ML_raw/train_dataset.geojson"
    gdf_train = load_geojson(file_path_tr)
    s1_tr, s2_tr, lab_tr = process_images_and_labels(root, gdf_train)

    file_path_val = "/home/giuseppe/Datasets/METER-ML_raw/val_dataset.geojson"
    gdf_val = load_geojson(file_path_val)
    s1_val, s2_val, lab_val = process_images_and_labels(root, gdf_val)

    file_path_test = "/home/giuseppe/Datasets/METER-ML_raw/test_dataset.geojson"
    gdf_test = load_geojson(file_path_test)
    s1_test, s2_test, lab_test = process_images_and_labels(root, gdf_test)

    s1 = np.concatenate((s1_tr, s1_val, s1_test), axis = 0)
    s2 = np.concatenate((s2_tr, s2_val, s2_test), axis = 0)
    label = np.concatenate((lab_tr, lab_val, lab_test), axis=0)

    s1 = z_normalization(s1)
    s2 = z_normalization(s2)

    np.save("/home/giuseppe/Datasets/METER-ML_raw/SAR_data_normalized.npy", s1)
    np.save("/home/giuseppe/Datasets/METER-ML_raw/MS_data_normalized.npy", s2)
    np.save("/home/giuseppe/Datasets/METER-ML_raw/labels.npy", label)
   

    '''

    file_path_val = "/home/giuseppe/Datasets/METER-ML_raw/val_dataset.geojson"
    gdf_val = load_geojson(file_path_val)
    s1_val, s2_val, lab_val = process_images_and_labels(root, gdf_val)

    file_path_test = "/home/giuseppe/Datasets/METER-ML_raw/test_dataset.geojson"
    gdf_test = load_geojson(file_path_test)
    s1_test, s2_test, lab_test = process_images_and_labels(root, gdf_test)

    s1 = np.concatenate((s1_val, s1_test), axis = 0)
    s2 = np.concatenate((s2_val, s2_test), axis = 0)
    label = np.concatenate((lab_val, lab_test), axis = 0)

    print(s2_test.shape)
    print(s2_val.shape)

    s1 = z_normalization(s1)
    s2 = z_normalization(s2)

    np.save("/home/giuseppe/Datasets/METER-ML_raw/SAR_data_normalized.npy", s1)
    np.save("/home/giuseppe/Datasets/METER-ML_raw/MS_data_normalized.npy", s2)
    np.save("/home/giuseppe/Datasets/METER-ML_raw/labels.npy", label)
    '''

