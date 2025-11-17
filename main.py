import sys
import os
from scipy import io
from param import ds, data_names, backbones_list
import SAHARA as sahara
import argparse

if __name__ == '__main__':
    # Creating Argument Parser
    parser = argparse.ArgumentParser(prog='SAHARA Training code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='',
                                     epilog='')

    # Required Arguments
    required = parser.add_argument_group('required named arguments')
    required.add_argument("-d", "--dataset", type=str, required=True, help='Name of the Dataset')
    required.add_argument("-s", "--source", type=str, required=True, help='Name of Source Modality')
    required.add_argument("-b", "--backbone", type=str, required=True, help='Name of the backbone architecture')

    # Optional Arguments
    optional = parser._action_groups.pop()
    optional.add_argument('-n_gpu', "--gpu_number", type=str, default=0,
                          help='Number of the GPU on which to run the algorithm.')
    optional.add_argument("-ns", "--nsamples", type=int, nargs="+", default=[5, 10, 25, 50],
                          help="List of numbers of labelled target samples")
    optional.add_argument("-np", "--nsplits", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="List of splits")
    optional.add_argument("-o", "--out_dir", type=str, default='Results',
                          help='The directory in which save the outcome.')
    optional.add_argument("-ds", "--ds_dir", type=str, default='Datasets',
                          help='The directory containing the datasets.')
    parser._action_groups.append(optional)

    # Arguments Parsing
    arguments = parser.parse_args()

    backbone = arguments.backbone
    gpu = arguments.gpu_number
    nsamples_list = arguments.nsamples
    nsplits_list = arguments.nsplits
    ds_dir = arguments.ds_dir

    # ---- Backbone Check ----
    if arguments.backbone not in backbones_list:
        raise ValueError(
            f"Backbone architecture '{arguments.backbone}' not found! Available: {backbones_list}")

    # ---- Dataset Check ----
    if arguments.dataset not in ds:
        raise ValueError(f"Dataset '{arguments.dataset}' not found! Available datasets: {ds}")
    ds_idx = ds.index(arguments.dataset)
    # os.path.is_file(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_data_filtered.npy"))

    # ---- Modality Check ----
    if arguments.source not in data_names[arguments.dataset]:
        raise ValueError(
            f"Source modality '{arguments.source}' not found for dataset '{arguments.dataset}'. Available: {data_names[arguments.dataset]}")
    source_idx = data_names[arguments.dataset].index(arguments.source)

    out_dir = os.path.join(arguments.out_dir, ds[ds_idx], backbone)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for nsamples in nsamples_list:
        table = []
        for nsplit in nsplits_list:
            print(
                f"SAHARA: Processing {arguments.dataset}. Source = {arguments.source}. Backbone = {arguments.backbone}. nsplit = {nsplit}, nsamples = {nsamples}")
            sys.stdout.flush()

            f1_val, f1_nw = sahara.train_and_eval(ds_dir, out_dir, nsamples, nsplit, ds_idx, source_idx, gpu, backbone)

            table.append([ds_idx, source_idx, nsplit, nsamples, f1_val, f1_nw])  # Save results

            io.savemat(
                os.path.join(out_dir, f'results_SAHARA_{arguments.dataset}_{arguments.source}_{arguments.backbone}_{nsamples}.mat'),
                {'RESULTS': table}
            )

'''
Example of istruction:
    python main.py -d RESISC45_EURO -s EURO -b CNN -n_gpu 0 -ds /home/user/Datasets
    python main.py -d RESISC45_EURO -s EURO -b CNN -n_gpu 0 -ns 5 10 25 50 -np 1 2 3 4 5 -ds /home/user/Datasets
'''