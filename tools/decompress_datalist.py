import numpy as np
import argparse
import time
from utils.compress_utils import BasicCompressor
from utils.utils import load_compressor_cfg
from utils.compress_utils import compress_point_cloud, decompress_point_cloud
from utils.segment_utils import PointCloudSegment
from utils.evaluate_metrics import calc_chamfer_distance, calc_point_to_point_plane_psnr
from dataset import build_dataset
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from utils.compress_utils import QuantizationModule, save_compressed_bitstream, read_compressed_bitstream
import IPython
import concurrent.futures as futures
from tqdm import tqdm

parser = argparse.ArgumentParser()
# Data related arguments
parser.add_argument('--datalist', help='datalist for batch of point cloud decompression.')
parser.add_argument('--output_dir', help='directory of the output files.')
parser.add_argument('--lidar', help='lidar type of this point cloud collection.')

parser.add_argument('--workers', type=int, default=1, help='number of workers for parallel compression')

parser.add_argument('--compressor_yaml', default=os.path.join(BASE_DIR, 'cfgs/compressor.yaml'))

parser.add_argument('--basic_compressor', type=str, default=None, help='for manual setting.')
parser.add_argument('--accuracy', type=float, default=None, help='for manual setting.')
parser.add_argument('--segment_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--cluster_num', type=int, default=None, help='for manual setting.')
parser.add_argument('--DBSCAN_eps', type=float, default=None, help='for manual setting.')
parser.add_argument('--model_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--angle_threshold', type=float, default=None, help='for manual setting.')
parser.add_argument('--nonuniform', action='store_true', help='for manual setting.')

parser.add_argument('--output', action='store_true', help='output the results on screen.')
parser.add_argument('--eval', action='store_true', help='evaluate the reconstruction quality.')

parser.add_argument('--cpu', action='store_true', help='set --cpu to make it run on CPU only if lack of GPU.')
args = parser.parse_args()
print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))


def compress():
    compressor_cfg = load_compressor_cfg(args.compressor_yaml)
    accuracy = compressor_cfg['accuracy'] * 2
    basic_compressor = BasicCompressor(compressor_yaml=args.compressor_yaml)
    dataset = build_dataset(lidar_type=args.lidar)

    segment_cfg = {
        'segment_method': compressor_cfg['segment_method'],
        'ground_vertical_threshold': compressor_cfg['ground_threshold'],
        'cluster_num': compressor_cfg['cluster_num'],  # used in farthest point sampling
        'DBSCAN_eps': compressor_cfg['DBSCAN_eps']  # used in DBSCAN
    }
    model_cfg = {
        'model_method': compressor_cfg['modeling_method'],
        'angle_threshold': compressor_cfg['plane_angle_threshold'],
    }

    if args.basic_compressor is not None:
        basic_compressor.set_method(args.basic_compressor)
    if args.accuracy is not None:
        accuracy = args.accuracy * 2
    if args.segment_method is not None:
        segment_cfg['segment_method'] = args.segment_method
    if args.cluster_num is not None:
        segment_cfg['cluster_num'] = args.cluster_num
    if args.DBSCAN_eps is not None:
        segment_cfg['DBSCAN_eps'] = args.DBSCAN_eps
    if args.model_method is not None:
        model_cfg['model_method'] = args.model_method
    if args.angle_threshold is not None:
        model_cfg['angle_threshold'] = args.angle_threshold

    model_num = segment_cfg['cluster_num'] + 1
    PCTransformer = dataset.PCTransformer

    if args.nonuniform:
        uniform = False
    else:
        if compressor_cfg['compress_framework'] == 'uniform':
            uniform = True
        else:
            uniform = False

    with open(args.datalist) as f:
        file_list = f.readlines()

    def decompress_datalist(file_path):
        file_path = file_path.strip()
        assert file_path.split('.')[-1] == 'rpcc'

        # reconstruct
        compressed_data = read_compressed_bitstream(file_path, uniform=uniform)
        residual_quantized, seg_idx, salience_level, plane_param = decompress_point_cloud(compressed_data,
                                                                                          basic_compressor,
                                                                                          model_num,
                                                                                          dataset.transform_map.shape[
                                                                                              0],
                                                                                          dataset.transform_map.shape[
                                                                                              1])
        if uniform:
            QM = QuantizationModule(accuracy)
        else:
            QM = QuantizationModule(accuracy, uniform=False,
                                    level_kp_num=tuple(compressor_cfg['level_key_point_num']),
                                    level_dacc=tuple(compressor_cfg['level_delta_acc']),
                                    ground_salience_level=compressor_cfg['ground_salience_level'],
                                    feature_region=compressor_cfg['feature_region'],
                                    segments=compressor_cfg['segments'],
                                    sharp_num=compressor_cfg['sharp_num'],
                                    less_sharp_num=compressor_cfg['less_sharp_num'],
                                    flat_num=compressor_cfg['flat_num'])
        residual = QM.dequantize_residual(residual_quantized, seg_idx, salience_level)

        pc_seg = PointCloudSegment(dataset.transform_map)
        range_image_pred = pc_seg.intra_predict(seg_idx, plane_param)
        range_image_rec = range_image_pred + residual
        point_cloud_rec = PCTransformer.range_image_to_point_cloud(range_image_rec)

        if file_path[0] == '/':
            file_path = file_path[1:]
        output_path = os.path.join(args.output_dir, file_path)
        output_path = output_path.replace(output_path.split('.')[-1], 'bin')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.save_point_cloud_to_file(output_path, point_cloud_rec)

    with futures.ThreadPoolExecutor(args.workers) as executor:
        list(tqdm(executor.map(decompress_datalist, file_list), total=len(file_list)))


if __name__ == '__main__':
    compress()
