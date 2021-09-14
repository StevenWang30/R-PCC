import IPython
import yaml
import copy
from easydict import EasyDict
import bz2
import gzip
import lz4  # lz4 version is 0.7.0
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from utils.contour_utils import ContourExtractor
from ops.cpp_modules import feature_extractor_cpp
from ops.cpp_modules import quantization_utils_cpp
import numpy as np
import struct


def extract_features(range_image, feature_region=3, segments=8, sharp_num=4, less_sharp_num=8, flat_num=6):
    feature_map, key_point_map = feature_extractor_cpp.extract_features(range_image, feature_region,
                                                                    segments, sharp_num, less_sharp_num, flat_num)
    return feature_map, key_point_map


class QuantizationModule:
    def __init__(self, accuracy, acc_deg=(0.06, 0.04, 0.02, 0), uniform=True):
        self.uniform = uniform
        self.acc_deg = acc_deg
        if uniform:
            self.acc = accuracy
        else:
            # non-keypoint, less feature, more feature, dense keypoint
            self.acc = [accuracy + self.acc_deg[0], accuracy + self.acc_deg[1], accuracy + self.acc_deg[2], accuracy]

    def quantize_residual(self, residual, seg_idx, point_cloud=None, range_image=None):
        if self.uniform:
            residual_quantized = quantization_utils_cpp.uniform_quantize(seg_idx, residual, self.acc)
            # # python version
            # residual_collect = []
            # cluster_num = np.max(seg_idx) + 1
            # for m in range(cluster_num):
            #     if m == 1:
            #         # zero points.
            #         continue
            #     residual_collect.extend(residual[..., 0][np.where(seg_idx == m)])
            # residual_quantized = np.rint(np.array(residual_collect) / self.acc).astype(np.int16)
            salience_score = None
            quantize_level = None
            key_point_map = None
        else:
            nonuniform_accuracy = self.acc
            # range_image, seg_idx, feature_region=3, segments=8, sharp_num=4, less_sharp_num=8, flat_num=6
            (curve_map, key_point_map) = feature_extractor_cpp.extract_features_with_segment(range_image, seg_idx, 3, 8, 4, 8, 6)

            cluster_num = np.max(seg_idx) + 1
            salience_score = np.zeros(cluster_num)
            quantize_level = np.zeros(cluster_num)

            for cluster_id in range(cluster_num):
                if cluster_id == 0 or cluster_id == 1:
                    continue
                cluster_idx = np.where(seg_idx == cluster_id)
                key_point_num = np.where(key_point_map[cluster_idx] > 0)[0].shape[0]
                cluster_points = point_cloud[cluster_idx]
                if cluster_points.shape[0] < 30:
                    continue
                if key_point_num > 3:
                    quantize_level[cluster_id] = 1
                if key_point_num > 10:
                    quantize_level[cluster_id] = 2
                if key_point_num > 30:
                    quantize_level[cluster_id] = 3
            quantize_level[0] = 1  # ground
            quantize_level[1] = 0  # zero points
            quantize_level = quantize_level.astype(np.uint8)
            residual_collect = []
            for m in range(cluster_num):
                if m == 1:
                    # zero points
                    continue
                cluster_acc = nonuniform_accuracy[quantize_level[m]]
                cur_residual = residual[..., 0][np.where(seg_idx == m)]
                residual_collect.extend(list(np.rint(cur_residual / cluster_acc)))

            residual_quantized = np.array(residual_collect)
        return residual_quantized, salience_score, quantize_level, key_point_map

    def dequantize_residual(self, quantized_residual, seg_idx, quantize_level=None):
        residual = np.zeros_like(seg_idx, dtype=np.float32)
        start = 0
        for m in range(seg_idx.max() + 1):
            idx = np.where(seg_idx == m)
            if m == 1:
                # zero points
                continue

            if self.uniform:
                cur_acc = self.acc
            else:
                cur_acc = self.acc[quantize_level[m]]
            residual[idx] = quantized_residual[start:start + idx[0].shape[0]] * cur_acc
            start += idx[0].shape[0]
        if start != quantized_residual.shape[0]:
            print('not correct.')
            IPython.embed()
        return np.expand_dims(residual, -1)


# numpy to bytes
# https://stackoverflow.com/questions/62352670/deserialization-of-large-numpy-arrays-using-pickle-is-order-of-magnitude-slower?noredirect=1#comment110277408_62352670
# https://stackoverflow.com/questions/53376786/convert-byte-array-back-to-numpy-array
def compress_point_cloud(basic_compressor, plane_param, cluster_idx, salience_level, nonzero_residual_quantized,
                         ground_residual_quantized=None, cluster_residual_quantized=None,
                         point_cloud=None, range_image=None, full=False):
    original_data = {}
    original_data['residual_quantized'] = nonzero_residual_quantized.astype(np.int16)

    if full:
        if point_cloud is not None:
            original_data['point_cloud'] = point_cloud.astype(np.float32)
        if range_image is not None:
            original_data['range_image'] = range_image.astype(np.float32)
        if ground_residual_quantized is not None:
            original_data['ground_residual'] = ground_residual_quantized.astype(np.int16)
        if cluster_residual_quantized is not None:
            original_data['cluster_residual'] = cluster_residual_quantized.astype(np.int16)

    if salience_level:
        original_data['salience_level'] = salience_level.astype(np.uint8)
    contour_map, idx_sequence = ContourExtractor.extract_contour(cluster_idx)
    contour_map = contour_map.astype(np.bool)
    contour_map = np.packbits(contour_map, axis=None)  # same as before
    original_data['contour_map'] = contour_map.astype(np.uint8)
    original_data['idx_sequence'] = idx_sequence.astype(np.uint16)
    original_data['plane_param'] = plane_param.astype(np.float32)

    compressed_data = basic_compressor.compress_dict(original_data)
    return original_data, compressed_data


def save_compressed_bitstream(file, compressed_data, uniform=True):
    with open(file, 'wb') as f:
        if not uniform:
            f.write(struct.pack('i', len(compressed_data['salience_level'])))
            f.write(compressed_data['salience_level'])
        f.write(struct.pack('i', len(compressed_data['contour_map'])))
        f.write(compressed_data['contour_map'])
        f.write(struct.pack('i', len(compressed_data['idx_sequence'])))
        f.write(compressed_data['idx_sequence'])
        f.write(struct.pack('i', len(compressed_data['plane_param'])))
        f.write(compressed_data['plane_param'])
        f.write(struct.pack('i', len(compressed_data['residual_quantized'])))
        f.write(compressed_data['residual_quantized'])


def read_compressed_bitstream(file, uniform=True):
    compressed_data = {}
    with open(file, "rb") as f:
        if not uniform:
            length = struct.unpack('i', f.read(4))[0]
            print(length)
            compressed_data['salience_level'] = f.read(length)
        length = struct.unpack('i', f.read(4))[0]
        compressed_data['contour_map'] = f.read(length)
        length = struct.unpack('i', f.read(4))[0]
        compressed_data['idx_sequence'] = f.read(length)
        length = struct.unpack('i', f.read(4))[0]
        compressed_data['plane_param'] = f.read(length)
        length = struct.unpack('i', f.read(4))[0]
        compressed_data['residual_quantized'] = f.read(length)
    return compressed_data


def decompress_point_cloud(compressed_data, basic_compressor, model_num, H, W):
    decompressed_data = basic_compressor.decompress_dict(compressed_data)
    plane_param = np.ndarray(shape=(model_num, 4), dtype=np.float32, buffer=decompressed_data['plane_param'])
    contour_map = np.ndarray(shape=(-1,), dtype=np.uint8, buffer=decompressed_data['contour_map'])
    contour_map = np.unpackbits(contour_map)
    contour_map = np.reshape(contour_map, (H, W))
    idx_sequence = np.ndarray(shape=(-1,), dtype=np.uint16, buffer=decompressed_data['idx_sequence'])
    idx_map = ContourExtractor.recover_map(contour_map, idx_sequence)

    if 'salience_level' in decompressed_data.keys():
        salience_level = np.ndarray(shape=(-1,), dtype=np.uint8, buffer=decompressed_data['salience_level'])
    else:
        salience_level = None
    residual_quantized = np.ndarray(shape=(-1, ), dtype=np.int16,
                                    buffer=decompressed_data['residual_quantized'])
    return residual_quantized, idx_map, salience_level, plane_param


def compress_plane_idx_map(plane_idx, single_line=True):
    if not single_line:
        from utils.contour_utils import ContourExtractorDoubleDirection
        contour_map, idx_sequence = ContourExtractorDoubleDirection.extract_contour(plane_idx)
        contour_map = contour_map.astype(np.bool)
        # for boolean data, packbits can save 1/8 (8 boolean to 1 uint8)
        contour_map = np.packbits(contour_map, axis=None)  # shape=(-1,), dtype=np.uint8
    else:
        from utils.contour_utils import ContourExtractor
        contour_map, idx_sequence = ContourExtractor.extract_contour(plane_idx)
        contour_map = contour_map.astype(np.bool)
        contour_map = np.packbits(contour_map, axis=None)  # same as before
    return contour_map, idx_sequence


class BasicCompressor:
    def __init__(self, compressor_yaml=None, method_name=None):
        self.method_name = None
        if compressor_yaml is not None:
            with open(compressor_yaml, 'r') as f:
                try:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    config = yaml.load(f)
            compressor_config = EasyDict(config)
            self.method_name = compressor_config.BASIC_COMPRESSOR_NAME
        if method_name is not None:
            self.method_name = method_name

        if self.method_name is not None:
            assert self.method_name in ['lz4', 'bzip2', 'gzip', 'deflate'], \
                'Compression method is not existed. (lz4, bzip2, gzip, deflate)'

    def set_method(self, method_name):
        self.method_name = method_name
        assert self.method_name in ['lz4', 'bzip2', 'gzip', 'deflate'], \
            'Compression method is not existed. (lz4, bzip2, gzip, deflate)'

    def compress_dict(self, data_dict):
        compressed_data_dict = copy.deepcopy(data_dict)
        for key, val in data_dict.items():
            # import time
            # t = time.time()
            compressed_data_dict[key] = self.compress(val)
            # print('compress ', key, ' time cost: ', time.time() - t)
        return compressed_data_dict

    def decompress_dict(self, data_dict):
        decompressed_data_dict = copy.deepcopy(data_dict)
        for key, val in data_dict.items():
            decompressed_data_dict[key] = self.decompress(val)
        return decompressed_data_dict

    def compress(self, np_array):
        if self.method_name == 'lz4':
            return self.lz4_compress(np_array)
        if self.method_name == 'bzip2':
            return self.bzip2_compress(np_array)
        if self.method_name == 'gzip' or self.method_name == 'deflate':
            return self.gzip_compress(np_array)

    def decompress(self, bitstream):
        if self.method_name == 'lz4':
            return self.lz4_decompress(bitstream)
        if self.method_name == 'bzip2':
            return self.bzip2_decompress(bitstream)
        if self.method_name == 'gzip' or self.method_name == 'deflate':
            return self.gzip_decompress(bitstream)

    def calc_compressed_bytes(self, np_array):
        compressed = self.compress(np_array)
        byte_len = len(compressed)
        return byte_len

    @staticmethod
    def lz4_compress(np_array):
        return lz4.dumps(np_array)

    @staticmethod
    def lz4_decompress(data):
        return lz4.loads(data)

    @staticmethod
    def bzip2_compress(np_array):
        return bz2.compress(np_array)

    @staticmethod
    def bzip2_decompress(data):
        return bz2.decompress(data)

    @staticmethod
    def gzip_compress(np_array):
        return gzip.compress(np_array)

    @staticmethod
    def gzip_decompress(data):
        return gzip.decompress(data)


if __name__ == "__main__":
    import numpy as np
    from time import time

    data_type = np.int8
    data_size = (64, 2000)
    rand_array = np.random.randint(50, size=data_size).astype(data_type)
    rand_bytes = rand_array.tobytes()
    repeat_time = 100

    # AE = BasicCompressor(method_name='arithmetic_coding')
    # compressed_size = AE.calc_compressed_bytes(rand_array)

    methods = ['lz4', 'bzip2', 'gzip']
    BC = BasicCompressor()
    for method in methods:
        print('\nTest ', method)
        BC.set_method(method)

        t0 = time()
        for i in range(repeat_time):
            compressed_data = BC.compress(rand_array)
        t1 = time()
        for i in range(repeat_time):
            decompressed_data = BC.decompress(compressed_data)
        print('%d times compress cost time: %.04f, decompress cost time: %.04f' % (repeat_time, t1 - t0, time() - t1))
        print('Compression rate: ', len(rand_bytes) / len(compressed_data))
        recovered = np.ndarray(shape=data_size, dtype=data_type, buffer=decompressed_data)
        assert np.array_equal(recovered, rand_array), '%s is not working.' % method
    print('All compression methods are working.')
