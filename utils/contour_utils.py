import IPython
import copy
import numpy as np


class FloodFill():
    def __init__(self, contour_map, original_idx_map=None, compresed_idx=None):
        self.idx_map = original_idx_map if original_idx_map is not None else None
        self.idx = compresed_idx if compresed_idx is not None else None
        self.contour_map = contour_map
        self.row = contour_map.shape[0]
        self.col = contour_map.shape[1]

    def sorted_index_encoder(self):
        '''
            To make sure each block has different plane id.
        '''
        assert self.idx_map is not None
        sorted_idx_map = copy.copy(self.idx_map)
        visit_flag = np.zeros((self.row, self.col))
        original_compressed_idx = []
        sorted_compressed_idx = []
        sorted_num = 1
        for r in range(self.row):
            for c in range(self.col):
                if visit_flag[r, c] == 0:
                    cur_idx = self.idx_map[r, c]
                    original_compressed_idx.append(cur_idx)
                    sorted_compressed_idx.append(sorted_num)
                    stack_list = []
                    stack_list.append((r, c))
                    while len(stack_list) != 0:
                        cur_r = stack_list[-1][0]
                        cur_c = stack_list[-1][1]
                        # print('cur_r: %d, cur_c: %d' % (cur_r, cur_c))
                        visit_flag[cur_r, cur_c] = 1
                        sorted_idx_map[cur_r, cur_c] = sorted_num
                        stack_list.remove(stack_list[-1])
                        # four neighbourhoods
                        if cur_r > 0 and visit_flag[cur_r - 1, cur_c] == 0 and self.contour_map[
                            cur_r - 1, cur_c, 1] == 0:
                            stack_list.append((cur_r - 1, cur_c))  # top
                        if cur_c > 0 and visit_flag[cur_r, cur_c - 1] == 0 and self.contour_map[
                            cur_r, cur_c - 1, 0] == 0:
                            stack_list.append((cur_r, cur_c - 1))  # left
                        if cur_r < self.row - 1 and visit_flag[cur_r + 1, cur_c] == 0 and self.contour_map[
                            cur_r, cur_c, 1] == 0:
                            stack_list.append((cur_r + 1, cur_c))  # bottom
                        if cur_c < self.col - 1 and visit_flag[cur_r, cur_c + 1] == 0 and self.contour_map[
                            cur_r, cur_c, 0] == 0:
                            stack_list.append((cur_r, cur_c + 1))  # right
                    sorted_num += 1

        return sorted_idx_map, sorted_compressed_idx, original_compressed_idx

    def index_encoder(self):
        assert self.idx_map is not None
        visit_flag = np.zeros((self.row, self.col))
        compressed_idx = []
        for r in range(self.row):
            for c in range(self.col):
                if visit_flag[r, c] == 0:
                    cur_idx = self.idx_map[r, c]
                    compressed_idx.append(cur_idx)
                    stack_list = []
                    stack_list.append((r, c))
                    while len(stack_list) != 0:
                        cur_r = stack_list[-1][0]
                        cur_c = stack_list[-1][1]
                        # print('cur_r: %d, cur_c: %d' % (cur_r, cur_c))
                        visit_flag[cur_r, cur_c] = 1
                        stack_list.remove(stack_list[-1])
                        # four neighbourhoods
                        if cur_r > 0 and visit_flag[cur_r - 1, cur_c] == 0 and self.contour_map[cur_r - 1, cur_c, 1] == 0:
                            stack_list.append((cur_r - 1, cur_c))  # top
                        if cur_c > 0 and visit_flag[cur_r, cur_c - 1] == 0 and self.contour_map[cur_r, cur_c - 1, 0] == 0:
                            stack_list.append((cur_r, cur_c - 1))  # left
                        if cur_r < self.row - 1 and visit_flag[cur_r + 1, cur_c] == 0 and self.contour_map[cur_r, cur_c, 1] == 0:
                            stack_list.append((cur_r + 1, cur_c))  # bottom
                        if cur_c < self.col - 1 and visit_flag[cur_r, cur_c + 1] == 0 and self.contour_map[cur_r, cur_c, 0] == 0:
                            stack_list.append((cur_r, cur_c + 1))  # right
        return np.array(compressed_idx)

    def index_decoder(self):
        assert self.idx is not None
        visit_flag = np.zeros((self.row, self.col))
        idx_map = np.zeros((self.row, self.col), np.int32)
        idx_num = 0
        for r in range(self.row):
            for c in range(self.col):
                if visit_flag[r, c] == 0:
                    cur_idx = self.idx[idx_num]
                    stack_list = []
                    stack_list.append((r, c))
                    while len(stack_list) != 0:
                        cur_r = stack_list[-1][0]
                        cur_c = stack_list[-1][1]
                        # print('cur_r: %d, cur_c: %d' % (cur_r, cur_c))
                        visit_flag[cur_r, cur_c] = 1
                        idx_map[cur_r, cur_c] = cur_idx
                        stack_list.remove(stack_list[-1])
                        # four neighbourhoods
                        if cur_r > 0 and visit_flag[cur_r - 1, cur_c] == 0 and self.contour_map[
                            cur_r - 1, cur_c, 1] == 0:
                            stack_list.append((cur_r - 1, cur_c))  # top
                        if cur_c > 0 and visit_flag[cur_r, cur_c - 1] == 0 and self.contour_map[
                            cur_r, cur_c - 1, 0] == 0:
                            stack_list.append((cur_r, cur_c - 1))  # left
                        if cur_r < self.row - 1 and visit_flag[cur_r + 1, cur_c] == 0 and self.contour_map[
                            cur_r, cur_c, 1] == 0:
                            stack_list.append((cur_r + 1, cur_c))  # bottom
                        if cur_c < self.col - 1 and visit_flag[cur_r, cur_c + 1] == 0 and self.contour_map[
                            cur_r, cur_c, 0] == 0:
                            stack_list.append((cur_r, cur_c + 1))  # right

                    idx_num += 1
        return idx_map


class ContourExtractorDoubleDirection:
    @staticmethod
    def extract_contour(idx_map):
        # get contour
        '''
            Example:
                idx_map:
                     1  1  1  1  2

                     3  2  2  1  2

                     3  2  1  1  2

                     3  3  2  2  2

                contour:
                       0  0  0  1  1
                     1  1  1  0  0
                       1  0  1  1  1
                     0  0  1  0  0
                       1  1  0  1  1
                     0  1  1  1  0
                       0  1  0  0  1
                     1  1  1  1  1
                contour_map:
                    (0, 1) (0, 1) (0, 1) (1, 0) (1, 0)
                    (1, 0) (0, 0) (1, 1) (1, 0) (1, 0)
                    (1, 0) (1, 1) (0, 1) (1, 1) (1, 0)
                    (0, 1) (1, 1) (0, 1) (0, 1) (1, 1)
        '''
        [row, col] = idx_map.shape
        contour_map = np.ones((row, col, 2))  # right and bottom contour
        row_dif = idx_map[1:, :] - idx_map[:-1, :]
        row_dif = np.append(row_dif, np.ones((1, col)), 0)
        bottom_contour = np.ones((row, col))
        bottom_contour[np.where(row_dif == 0)] = 0
        col_dif = idx_map[:, 1:] - idx_map[:, :-1]
        col_dif = np.append(col_dif, np.ones((row, 1)), 1)
        right_contour = np.ones((row, col))
        right_contour[np.where(col_dif == 0)] = 0
        contour_map[:, :, 0] = right_contour
        contour_map[:, :, 1] = bottom_contour

        # use contour get compressed index sequence
        FF = FloodFill(contour_map, original_idx_map=idx_map)
        idx_sequence = FF.index_encoder()

        return contour_map, idx_sequence

    @staticmethod
    def recover_map(contour_map, idx_sequence):
        FF = FloodFill(contour_map, compresed_idx=idx_sequence)
        idx_map = FF.index_decoder()
        return idx_map


class ContourExtractor:
    @staticmethod
    def extract_contour(idx_map):
        '''
        Example:
            idx_map:
                 1  1  1  1  2
                 3  2  2  1  2
                 3  2  1  1  2
                 3  3  2  2  2

            contour:
                1  0  0  0  1
                1  1  0  1  1
                1  1  1  0  1
                1  0  1  0  0
            idx_sequence:
                1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2
        '''
        [row, col] = idx_map.shape
        contour_map = np.zeros((row, col))
        contour_map[:, 0] = 1
        contour_map[:, 1:] = idx_map[:, 1:] - idx_map[:, :-1]
        contour_map[np.where(contour_map != 0)] = 1
        idx_sequence = idx_map[np.where(contour_map == 1)]
        return contour_map, idx_sequence

    @staticmethod
    def recover_map(contour_map, idx_sequence):
        idx_map = np.zeros_like(contour_map, dtype=np.uint16)
        cm_flat = contour_map.reshape(-1)
        idx_map = idx_map.reshape(-1)
        pointer = 0
        for i in range(idx_sequence.shape[0]):
            index = idx_sequence[i]
            idx_map[pointer] = index
            pointer += 1
            if pointer >= cm_flat.shape[0]:
                break
            while cm_flat[pointer] == 0:
                idx_map[pointer] = index
                pointer += 1
                if pointer >= cm_flat.shape[0]:
                    break
        return idx_map.reshape(contour_map.shape)


if __name__ == '__main__':
    # idx = np.loadtxt('data/plane_idx.txt')
    # idx = idx[0].astype(np.uint8)
    # idx = idx.reshape((64, 2000))
    # ori_size = get_lz4_compressed_size(idx)
    # idx = np.array([
    #     [1, 1, 1, 1, 2],
    #     [3, 2, 2, 1, 2],
    #     [3, 2, 1, 1, 2],
    #     [3, 3, 2, 2, 2],
    # ])

    idx = np.array([
        [1, 1, 1, 1, 2, 1, 3, 4, 4],
        [3, 2, 2, 1, 2, 1, 1, 3, 4],
        [3, 2, 1, 1, 2, 4, 4, 3, 4],
        [3, 3, 2, 2, 2, 1, 4, 4, 4],
    ])

    cm, idx_seq = ContourExtractor.extract_contour(idx)
    print('contour map: \n', cm)
    print('idx sequence: \n', idx_seq)
    idx_rec = ContourExtractor.recover_map(cm, idx_seq)
    print('idx recovered: \n', idx_rec)

    # # for FloodFill contour map and compression test
    # cm = get_contour_from_idx_map(idx)
    # FF = FloodFill(cm, original_idx_map=idx)
    # # sorted_idx_map, sorted_compressed_idx, original_compressed_idx = FF.sorted_index_encoder()
    # compressed_idx = FF.index_encoder()
    #
    # # contour_size = get_lz4_compressed_size(cm.astype(np.int8)) // 8
    # # idx_size = get_lz4_compressed_size(np.array(compressed_idx))
    # #
    # # bin_cm = 'log_train/test/contour.npy'
    # # np.save(bin_cm, cm.astype(np.bool))
    # #
    # # bin_compressed_idx_map = 'log_train/test/compressed_idx.npy'
    # # np.save(bin_compressed_idx_map, np.array(compressed_idx).astype(np.uint8))
    #
    # FF_d = FloodFill(cm, compresed_idx=compressed_idx)
    # back = FF_d.index_decoder()
    IPython.embed()