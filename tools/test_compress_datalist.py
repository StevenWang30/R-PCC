import numpy as np
import argparse
from datasets import build_dataset
import time
import IPython
from utils.utils import np_size, sys_size, bit_size
from utils.compress_utils import BasicCompressor, compress_plane_idx_map
from utils.compress_utils import compress_point_cloud, decompress_point_cloud, load_compressor_cfg
from utils.visualize_utils import save_point_cloud_to_pcd, compare_point_clouds
import copy
from utils.compress_utils import compress_point_cloud, decompress_point_cloud
from utils.segment_utils import PointCloudSegment
from utils.evaluate_metrics import calc_chamfer_distance
from utils.compress_utils import QuantizationModule, extract_features
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


parser = argparse.ArgumentParser()
# Path related arguments
parser.add_argument('--result_dir', default=os.path.join(BASE_DIR, 'experiment_results/temp_results'))
# Data related arguments
# parser.add_argument('--datalist', default='data/test_64E_KITTI_city.txt')
parser.add_argument('--datalist', default=os.path.join(BASE_DIR, 'data/test_64E_KITTI_city_unsync.txt'))
parser.add_argument('--dataset', default='KITTI')
parser.add_argument('--compressor_yaml', default=os.path.join(BASE_DIR, 'cfgs/compressor.yaml'))

parser.add_argument('--basic_compressor', type=str, default=None, help='for manual setting.')
parser.add_argument('--accuracy', type=float, default=None, help='for manual setting.')
parser.add_argument('--segment_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--cluster_num', type=int, default=None, help='for manual setting.')
parser.add_argument('--DBSCAN_eps', type=float, default=None, help='for manual setting.')
parser.add_argument('--model_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--angle_threshold', type=float, default=None, help='for manual setting.')
parser.add_argument('--nonuniform', action='store_true', help='for manual setting.')

parser.add_argument('--max_eval_frames', type=int, default=-1, help='Only eval first N frames. When it == -1, eval all.')

# parser.add_argument('--eval_chamfer_distance', action='store_true')
#
parser.add_argument('--output_file', default=None)

parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))

scene = args.datalist.split('/')[-1]


def main():
    compressor_cfg = load_compressor_cfg(args.compressor_yaml)
    accuracy = compressor_cfg['ACCURACY'] * 2
    basic_compressor = BasicCompressor(compressor_yaml=args.compressor_yaml)
    test_dataset = build_dataset(datalist=args.datalist, dataset_name=args.dataset)
    pc_seg = PointCloudSegment(test_dataset.transform_map)
    segment_cfg = {
        'segment_method': compressor_cfg['SEGMENT_METHOD'],
        'ground_vertical_threshold': compressor_cfg['GROUND_THRESHOLD'],
        'cluster_num': compressor_cfg['CLUSTER_NUM'],  # used in farthest point sampling
        'DBSCAN_eps': compressor_cfg['DBSCAN_EPS']  # used in DBSCAN
    }
    model_cfg = {
        'model_method': compressor_cfg['MODEL_METHOD'],
        'angle_threshold': compressor_cfg['PLANE_ANGLE_THRESHOLD'],
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

    PCTransformer = test_dataset.PCTransformer

    compressed_size_list = []
    compressed_size_base_pc_list = []
    compressed_size_base_ri_list = []
    original_size_list = []
    point_num_list = []
    residual_size_list = []
    info_size_list = []
    residual_list = []
    for i in range(len(test_dataset)):
        if args.max_eval_frames is not None and args.max_eval_frames != -1:
            if i >= args.max_eval_frames:
                break

        point_cloud, range_image, original_point_cloud, file_name = test_dataset[i]

        if args.dataset == 'KITTI':
            if np.where(original_point_cloud[..., 0] != 0)[0].shape[0] < 0.8 * range_image.shape[0] * range_image.shape[1]:
                # for some KITTI samples, the points are preprocessed, most of ground points are deleted.
                continue

        nonzero_idx = np.where(range_image[..., 0] != 0)
        point_num = nonzero_idx[0].shape[0]

        # plane_param, residual, seg_idx = pc_seg.segment(point_cloud, range_image)
        seg_idx, ground_model = pc_seg.segment(point_cloud, range_image, segment_cfg, cpu=args.cpu)
        cluster_models = pc_seg.cluster_modeling(point_cloud, range_image, seg_idx, model_cfg)
        model_param = np.concatenate((ground_model.reshape(1, 4), cluster_models), 0)
        range_image_pred = pc_seg.intra_predict(seg_idx, model_param)
        # point_cloud_pred = PCTransformer.range_image_to_point_cloud(range_image_pred)
        residual = range_image - range_image_pred
        residual_list.append(np.abs(residual).sum() / point_num)

        model_num = seg_idx.max() + 1

        QM = QuantizationModule(accuracy, uniform=(not args.nonuniform))
        residual_quantized, salience_score, quantize_level, key_point_map = QM.quantize_residual(residual,
                                                                                                 seg_idx,
                                                                                                 point_cloud,
                                                                                                 range_image)

        # start compress
        # nonzero_residual_quantized = np.rint(residual[nonzero_idx] / accuracy)
        original_data, compressed_data = compress_point_cloud(basic_compressor, model_param,
                                                              seg_idx, residual_quantized)

        # only basic compressor
        nonzero_point_cloud_quantized = np.rint(point_cloud[nonzero_idx] / accuracy).astype(np.int16)
        compressed_size_base_pc_list.append(len(basic_compressor.compress(nonzero_point_cloud_quantized)))

        range_image_quantized_base = np.rint(range_image / accuracy).astype(np.int16)
        compressed_size_base_ri_list.append(len(basic_compressor.compress(range_image_quantized_base)))

        print('\ncompose ', file_name)
        print('size of each part in original and compressed data:')
        for key, val in original_data.items():
            print(key, 'original: ', np_size(val), ', compressed: ', bit_size(compressed_data[key]))

        bit_stream = compressed_data['residual_quantized'] + \
                     compressed_data['contour_map'] + \
                     compressed_data['idx_sequence'] + \
                     compressed_data['plane_param']

        cr = np_size(point_cloud[nonzero_idx]) / bit_size(bit_stream)
        print('Ours: \n'
              '    CR: %.2f, bpp: %.2f, (residual %.1f%%, contour_map %.1f%%, idx_sequence %.1f%%, cluster_param %.1f%%)'
              % (cr, (len(bit_stream) * 8) / point_num, len(compressed_data['residual_quantized']) / len(bit_stream) * 100
                 , len(compressed_data['contour_map']) / len(bit_stream) * 100
                 , len(compressed_data['idx_sequence']) / len(bit_stream) * 100
                 , len(compressed_data['plane_param']) / len(bit_stream) * 100))

        compressed_size_list.append(bit_size(bit_stream))
        original_size_list.append(np_size(point_cloud[nonzero_idx]))
        info_size_list.append(bit_size(compressed_data['contour_map'] + compressed_data['idx_sequence'] +
                                    compressed_data['plane_param']))
        residual_size_list.append(bit_size(compressed_data['residual_quantized']))
        point_num_list.append(point_num)

        print('Only basic compressor for range image: \n'
              '    CR: %.2f, bpp: %.2f' %
              (original_size_list[-1] / compressed_size_base_ri_list[-1],
               compressed_size_base_ri_list[-1] * 8 / point_num))
        print('Only basic compressor for point cloud:\n'
              '    CR: %.2f, bpp: %.2f' %
              (original_size_list[-1] / compressed_size_base_pc_list[-1],
               compressed_size_base_pc_list[-1] * 8 / point_num))

        # if cr < 20:
        #     color = COLOR[seg_idx]
        #     save_point_cloud_to_pcd(point_cloud, color=color, save=False, vis=True)
        #     # compare_point_clouds(point_cloud, original_point_cloud, vis_all=True, save=False, vis=True)
        #     IPython.embed()

    print('\n\nTest finished.')
    frame_num = len(original_size_list)
    compression_rate_list = np.array(original_size_list) / np.array(compressed_size_list)
    bpp_list = np.array(compressed_size_list) * 8 / np.array(point_num_list)
    info_bpp_list = np.array(info_size_list) * 8 / np.array(point_num_list)
    residual_bpp_list = np.array(residual_size_list) * 8 / np.array(point_num_list)
    baseline_compression_rate_pc_list = np.array(original_size_list) / np.array(compressed_size_base_pc_list)
    baseline_bpp_pc_list = np.array(compressed_size_base_pc_list) * 8 / np.array(point_num_list)
    baseline_compression_rate_ri_list = np.array(original_size_list) / np.array(compressed_size_base_ri_list)
    baseline_bpp_ri_list = np.array(compressed_size_base_ri_list) * 8 / np.array(point_num_list)
    print('In scene %s, test %d point cloud frames, when accuracy is %.4f, the mean compression rate is: %.2f, the mean bpp is: %.2f' %
          (scene, frame_num, accuracy / 2, compression_rate_list.mean(), bpp_list.mean()))
    print('mean of info bpp: %.2f, mean of residual bpp: %.2f' % (info_bpp_list.mean(), residual_bpp_list.mean()))
    print('\nOnly basic compressor: ')
    print('The mean compression rate is: %.2f, the mean bpp is: %.2f' %
          (baseline_compression_rate_ri_list.mean(), baseline_bpp_ri_list.mean()))

    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "a+") as f:
            f.write('Evaluation parameters...\n')
            for key, val in vars(args).items():
                f.write("{:16} {}".format(key, val))
                f.write('\n')
            f.write('\nAll data evaluation finished. \n')
            f.write('Scene: %s\n' % scene)
            f.write('Test %d point cloud frames\n' % frame_num)
            f.write('Accuracy: %.4f\n' % (accuracy / 2))
            f.write('Result:\n')
            f.write('    Ours method: (min/max/mean)\n')
            f.write('        compression rate: (%.2f/%.2f/%.2f)\n'
                    '        bpp: (%.2f/%.2f/%.2f)\n'
                    '        info bpp: (%.2f/%.2f/%.2f)\n'
                    '        residual bpp: (%.2f/%.2f/%.2f)\n'
                    '        residual: (%.2f/%.2f/%.2f)\n'
                    %
                    (compression_rate_list.min(), compression_rate_list.max(), compression_rate_list.mean(),
                     bpp_list.min(), bpp_list.max(), bpp_list.mean(),
                     info_bpp_list.min(), info_bpp_list.max(), info_bpp_list.mean(),
                     residual_bpp_list.min(), residual_bpp_list.max(), residual_bpp_list.mean(),
                     np.array(residual_list).min(), np.array(residual_list).max(), np.array(residual_list).mean()))

            f.write('\n    Only basic compressor for point cloud: \n')
            f.write('        compression rate: (%.2f/%.2f/%.2f)\n'
                    '        bpp: (%.2f/%.2f/%.2f)\n'
                    %
                    (baseline_compression_rate_pc_list.min(), baseline_compression_rate_pc_list.max(), baseline_compression_rate_pc_list.mean(),
                     baseline_bpp_pc_list.min(), baseline_bpp_pc_list.max(), baseline_bpp_pc_list.mean()))

            f.write('\n    Only basic compressor for range image: \n')
            f.write('        compression rate: (%.2f/%.2f/%.2f)\n'
                    '        bpp: (%.2f/%.2f/%.2f)\n'
                    %
                    (baseline_compression_rate_ri_list.min(), baseline_compression_rate_ri_list.max(),
                     baseline_compression_rate_ri_list.mean(),
                     baseline_bpp_ri_list.min(), baseline_bpp_ri_list.max(), baseline_bpp_ri_list.mean()))


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__))))

    main()
