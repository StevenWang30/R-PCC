import open3d as o3d
import numpy as np
import IPython
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib import colors


def o3d_draw_pcd(o3d_pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    print('Press K to change background color...')
    print('Press -/+ to change point size...')
    o3d.visualization.draw_geometries_with_key_callbacks([o3d_pcd], key_to_callback)


def save_point_cloud_to_pcd(pc_data, save_path=None, color=None, save=True, vis=False, output=True):
    idx = np.where(np.linalg.norm(pc_data, ord=2, axis=-1) != 0)
    pc_vec = pc_data[idx]
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(pc_vec)
    if color is not None:
        color_vec = color[idx]
        points_o3d.colors = o3d.utility.Vector3dVector(color_vec)
    else:
        points_o3d.paint_uniform_color([1, 0, 0])
    if save:
        if save_path is not None:
            if output:
                print('write pcd file into ', save_path)
            o3d.io.write_point_cloud(save_path, points_o3d)
    if vis:
        # o3d.visualization.draw_geometries([points_o3d])
        o3d_draw_pcd(points_o3d)


def compare_point_clouds(pc1, pc2, vis_all=False, save_path=None, save=True, vis=False, output=True):
    pc1 = pc1[np.where(np.sum(pc1, -1) != 0)]
    pc2 = pc2[np.where(np.sum(pc2, -1) != 0)]
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    dist_pc2_to_pc1 = np.asarray(pc2_o3d.compute_point_cloud_distance(pc1_o3d))
    print('chamfer distance pc2 to pc1: max-', dist_pc2_to_pc1.max(), ', min-', dist_pc2_to_pc1.min(), ', mean-', dist_pc2_to_pc1.mean())
    # # dist_pc1_to_pc2 = np.asarray(pc1_o3d.compute_point_cloud_distance(pc2_o3d))
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
    norm = matplotlib.colors.Normalize(vmin=np.min(dist_pc2_to_pc1), vmax=np.max(dist_pc2_to_pc1))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)  # 'Reds'
    pc2_colors = mapper.to_rgba(dist_pc2_to_pc1)[:, :-1]
    # pc2_colors = np.ones_like(pc2_colors) * [1, 0, 0]
    pc2_o3d.colors = o3d.utility.Vector3dVector(pc2_colors)
    if vis_all:
        pc1_colors = np.ones_like(pc1)
        pc1_colors[:, :] = [0.7, 1, 1]  # [0.7, 1, 1]#[0.7, 0.7, 0.7]
        # pc2_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(pc2_colors) * [0, 0, 1])
        pc1_o3d.colors = o3d.utility.Vector3dVector(pc1_colors)
        pc2_o3d = pc1_o3d + pc2_o3d

    if save:
        if save_path is not None:
            if output:
                print('write pcd file into ', save_path)
            o3d.io.write_point_cloud(save_path, pc2_o3d)
    if vis:
        # o3d.visualization.draw_geometries([pc2_o3d])
        o3d_draw_pcd(pc2_o3d)


def draw_qualitative_point_clouds(pc1, pc2, vis_all=False, save_path=None, save=True, vis=False, output=True):
    pc1 = pc1[np.where(np.sum(pc1, -1) != 0)]
    pc2 = pc2[np.where(np.sum(pc2, -1) != 0)]
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    dist_pc1_to_pc2 = np.asarray(pc1_o3d.compute_point_cloud_distance(pc2_o3d))
    print('chamfer distance pc1 to pc2: max-', dist_pc1_to_pc2.max(), ', min-', dist_pc1_to_pc2.min(), ', mean-',
          dist_pc1_to_pc2.mean())
    dist_pc2_to_pc1 = np.asarray(pc2_o3d.compute_point_cloud_distance(pc1_o3d))
    print('chamfer distance pc2 to pc1: max-', dist_pc2_to_pc1.max(), ', min-', dist_pc2_to_pc1.min(), ', mean-',
          dist_pc2_to_pc1.mean())
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)  # 'Reds'
    pc1_colors = mapper.to_rgba(dist_pc1_to_pc2)[:, :-1]
    # pc2_colors = np.ones_like(pc2_colors) * [1, 0, 0]
    pc1_o3d.colors = o3d.utility.Vector3dVector(pc1_colors)
    if vis_all:
        pc1_colors = np.ones_like(pc1)
        pc1_colors[:, :] = [0, 1, 0]#[0.7, 1, 1]#[0.7, 0.7, 0.7]
        pc2_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(pc2) * [1, 0, 0])
        pc1_o3d.colors = o3d.utility.Vector3dVector(pc1_colors)
        pc1_o3d = pc1_o3d + pc2_o3d

    if save:
        if save_path is not None:
            if output:
                print('write pcd file into ', save_path)
            o3d.io.write_point_cloud(save_path, pc1_o3d)
    if vis:
        # o3d.visualization.draw_geometries([pc2_o3d])
        o3d_draw_pcd(pc1_o3d)



def visualize_left_points(pc1, pc2, save_path=None, save=True, vis=False, output=True):
    from utils.evaluate_metrics import calc_chamfer_distance
    result = calc_chamfer_distance(pc1, pc2)

    pc1_colors = np.ones_like(pc1)
    pc1_colors[:, :] = [1, 0, 0]
    pc1_colors[result['chamfer_dist_info']['idx2']] = [0.7, 1, 1]
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc1_o3d.colors = o3d.utility.Vector3dVector(pc1_colors)
    if output and save and save_path is not None:
        print('write pcd file into ', save_path)
    if save:
        o3d.io.write_point_cloud(save_path, pc1_o3d)
    if vis:
        o3d.visualization.draw_geometries([pc1_o3d])


def visualize_plane_range_image(plane_idx, save_path=None, pixel_distance=None, threshold=999):
    if pixel_distance is not None:
        distance = pixel_distance[0].reshape((16, 1800))
    idx = plane_idx[0].reshape((16, 1800))
    color = np.random.random((np.max(idx) + 1, 3))
    ri_image = color[idx]
    # invalid_mask = np.where(distance > threshold)
    # ri_image[invalid_mask] = [1, 0, 0]
    imageio.imwrite(save_path, ri_image)


def visualize_contour_map(range_image, plane_idx, save_path):
    from utils.contour_utils import ContourExtractorDoubleDirection
    contour_map, idx_seq = ContourExtractorDoubleDirection.extract_contour(plane_idx)
    IPython.embed()
    contour_img = np.zeros((range_image.shape[0]*2+1, range_image.shape[1]*2+1))
    contour_img[1::2, 1::2] = range_image[:, :, 0] / np.max(range_image)
    contour_img[0, :] = 1
    contour_img[:, 0] = 1
    contour_img[1::2, 2::2] = contour_map[:, :, 0]
    contour_img[2::2, 1::2] = contour_map[:, :, 1]

    imageio.imwrite(save_path, contour_img)


def visualize_index_map(idx_map):
    colors = np.random.random((idx_map.max() + 1, 3))
    img = colors[idx_map]
    plt.imshow(img)


def visualize_points_vertical_angle_distribution(points):
    vertical_angle = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], 2, -1))
    vertical_angle = np.degrees(vertical_angle)
    plt.hist(vertical_angle, bins=500)
    x_ticks = np.arange(vertical_angle.min(), vertical_angle.max(), (vertical_angle.max() - vertical_angle.min()) / 50)
    plt.xticks(x_ticks)
    plt.show()


def visualize_key_point_map(point_cloud, key_point_map, save_path=None, save=False, vis=True, output=True):
    color = np.ones_like(point_cloud) * 0.3
    color[np.where(key_point_map[..., 0] == 1)] = [1, 0, 0]
    color[np.where(key_point_map[..., 0] == 2)] = [0, 1, 0]
    color[np.where(key_point_map[..., 0] == 3)] = [0, 0, 1]

    valid_idx = np.where(point_cloud[..., 0] != 0)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(point_cloud[valid_idx])
    pc_o3d.colors = o3d.utility.Vector3dVector(color[valid_idx])
    if output and save and save_path is not None:
        print('write pcd file into ', save_path)
    if save:
        o3d.io.write_point_cloud(save_path, pc_o3d)
    if vis:
        o3d_draw_pcd(pc_o3d)
