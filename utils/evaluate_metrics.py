import numpy as np
import IPython
from scipy.spatial import cKDTree
from numba import njit
import open3d as o3d
import time
import torch

def calc_chamfer_distance(points1, points2, f1_threshold=0.02, out=True):
    from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
    from utils.ChamferDistancePytorch import fscore

    t = time.time()
    pc_1 = points1[np.where(np.sum(points1, -1) != 0)]
    pc_2 = points2[np.where(np.sum(points2, -1) != 0)]
    pc_1 = torch.from_numpy(pc_1).unsqueeze(0).cuda().float()  # 98512 * 3
    pc_2 = torch.from_numpy(pc_2).unsqueeze(0).cuda().float()  # 98512 * 3
    chamDist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chamDist(pc_1, pc_2)
    f_score, precision, recall = fscore.fscore(dist1, dist2, threshold=f1_threshold ** 2)
    cham_dist1 = torch.sqrt(dist1).mean().item()
    cham_dist2 = torch.sqrt(dist2).mean().item()

    result = {
        'max': max(cham_dist1, cham_dist2),
        'mean': (cham_dist1 + cham_dist2) / 2,
        'sum': cham_dist1 + cham_dist2,
        'cd1': cham_dist1,
        'cd2': cham_dist2,
        'f_score': f_score.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'chamfer_dist_info': {
            'dist1': dist1.cpu().numpy()[0],
            'dist2': dist2.cpu().numpy()[0],
            'idx1': idx1.cpu().numpy()[0],
            'idx2': idx2.cpu().numpy()[0],
        }
    }

    if out:
        for key, value in result.items():
            print(key, value)
        print('time cost: ', time.time() - t)
    return result


def calc_point_to_point_plane_psnr(points1, points2, idx1=None, idx2=None, r=59.7, out=True):
    t = time.time()
    pc_1 = points1[np.where(np.sum(points1, -1) != 0)]
    pc_2 = points2[np.where(np.sum(points2, -1) != 0)]
    if idx1 is None:
        t1 = cKDTree(pc_1, balanced_tree=False)
        _, idx1 = t1.query(pc_2, n_jobs=-1)
    if idx2 is None:
        t2 = cKDTree(pc_2, balanced_tree=False)
        _, idx2 = t2.query(pc_1, n_jobs=-1)

    max_energy = 3 * r * r
    pc_1_ngb = pc_2[idx2]
    pc_2_ngb = pc_1[idx1]
    point_to_point_mse_1 = np.sum(np.sum((pc_1 - pc_1_ngb) ** 2, axis=1)) / pc_1.shape[0]
    point_to_point_mse_2 = np.sum(np.sum((pc_2 - pc_2_ngb) ** 2, axis=1)) / pc_2.shape[0]
    point_to_point_psnr_1 = psnr(point_to_point_mse_1, max_energy)
    point_to_point_psnr_2 = psnr(point_to_point_mse_2, max_energy)
    point_to_point_result = {
        'psnr_1': point_to_point_psnr_1,
        'psnr_2': point_to_point_psnr_2,
        'mse_1': point_to_point_mse_1,
        'mse_2': point_to_point_mse_2,
        'psnr_mean': (point_to_point_psnr_1 + point_to_point_psnr_2) / 2,
        'mse_mean': (point_to_point_mse_1 + point_to_point_mse_2) / 2,
    }

    pc_1_n = compute_point_cloud_normal(pc_1)
    # pc_2_n = compute_point_cloud_normal(pc_2)
    # Compute normals in pc_2 from normals in pc_1
    pc_2_n = assign_attr(pc_1_n, idx1, idx2)
    pc_1_ngb_n = pc_2_n[idx2]
    pc_2_ngb_n = pc_1_n[idx1]
    # D2 may not exactly match mpeg-pcc-dmetric because of variations in nearest neighbors chosen when at equal distances
    point_to_plane_mse_1 = np.sum(np.sum((pc_1 - pc_1_ngb) * pc_1_ngb_n, axis=1) ** 2) / pc_1.shape[0]
    point_to_plane_mse_2 = np.sum(np.sum((pc_2 - pc_2_ngb) * pc_2_ngb_n, axis=1) ** 2) / pc_2.shape[0]
    point_to_plane_psnr_1 = psnr(point_to_plane_mse_1, max_energy)
    point_to_plane_psnr_2 = psnr(point_to_plane_mse_2, max_energy)
    point_to_plane_result = {
        'psnr_1': point_to_plane_psnr_1,
        'psnr_2': point_to_plane_psnr_2,
        'mse_1': point_to_plane_mse_1,
        'mse_2': point_to_plane_mse_2,
        'psnr_mean': (point_to_plane_psnr_1 + point_to_plane_psnr_2) / 2,
        'mse_mean': (point_to_plane_mse_1 + point_to_plane_mse_2) / 2,
    }
    if out:
        print('point_to_point_result: ')
        for key, value in point_to_point_result.items():
            print(key, value)
        print('point_to_plane_result: ')
        for key, value in point_to_plane_result.items():
            print(key, value)

        print('time cost: ', time.time() - t)
    return point_to_point_result, point_to_plane_result


@njit
def assign_attr(attr1, idx1, idx2):
    """Given point sets x1 and x2, transfers attributes attr1 from x1 to x2.
    idx1: N2 array containing the nearest neighbors indices of x2 in x1
    idx2: N1 array containing the nearest neighbors indices of x1 in x2
    """
    counts = np.zeros(idx1.shape[0])
    attr_sums = np.zeros((idx1.shape[0], attr1.shape[1]))
    for i, idx in enumerate(idx2):
        counts[idx] += 1
        attr_sums[idx] += attr1[i]
    for i, idx in enumerate(idx1):
        if counts[i] == 0:
            counts[i] += 1
            attr_sums[i] += attr1[idx]
    counts = np.expand_dims(counts, -1)
    attr2 = attr_sums / counts
    return attr2


def psnr(x, max_energy):
    return 10 * np.log10(max_energy / x)


def compute_point_cloud_normal(points):
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points)
    points_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=59.7, max_nn=12))
    # o3d.visualization.draw_geometries([points_o3d], point_show_normal=True)
    normal = np.asarray(points_o3d.normals)
    return normal


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    pc1 = np.random.random((200000, 3))
    pc2 = np.random.random((200000, 3))
    cd_result = calc_chamfer_distance(pc1, pc2)
    p2point_result, p2plane_result = calc_point_to_point_plane_psnr(pc1, pc2, idx1=cd_result['chamfer_dist_info']['idx1'], idx2=cd_result['chamfer_dist_info']['idx2'])

    IPython.embed()
