#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>

namespace py = pybind11;
using namespace std;


bool mark_as_picked(float * range_image_ptr, vector<vector<int>> &cloud_neighbors_picked, int h_i, int w_i, int feature_region){
    float near_threshold = 0.2, gap_threshold = 0.3;
    int w = cloud_neighbors_picked[0].size();
    bool ret_flag = true;

    float r = range_image_ptr[h_i * w + w_i];
    for (int i = -feature_region; i <= feature_region; i++){
        float r_neighbor = range_image_ptr[h_i * w + w_i + i];
        float dif = r - r_neighbor;
        if (abs(dif) < near_threshold)
            cloud_neighbors_picked[h_i][w_i] = 1;
        if (dif > gap_threshold)
            ret_flag = false;
    }
    return ret_flag;
}


std::tuple<py::array_t<float>, py::array_t<int>> extract_features_with_segment(py::array_t<float> range_image, py::array_t<int> seg_idx, int feature_region, int segments, int sharp_num, int less_sharp_num, int flat_num) {
    // input array
    py::buffer_info ri_buf = range_image.request();
    py::buffer_info seg_idx_buf = seg_idx.request();
    float *ri_ptr = (float *) ri_buf.ptr;
    int *seg_idx_ptr = (int *) seg_idx_buf.ptr;
    int h = ri_buf.shape[0];
    int w = ri_buf.shape[1];

    // return array
    py::array_t<float> sharp_feat_map = py::array_t<float>(ri_buf.size);
    py::array_t<int> key_point_map = py::array_t<int>(ri_buf.size);
    py::buffer_info feat_buf = sharp_feat_map.request();
    py::buffer_info kp_buf = key_point_map.request();
    float *feat_ptr = (float *) feat_buf.ptr;
    int *kp_ptr = (int *) kp_buf.ptr;

    vector<vector<int>> cloud_neighbors_picked(h, vector<int>(w, 0));

    for (int h_i = 0; h_i < h; h_i++) {
        vector<float> valid_ri;
        vector<int> valid_idx;

        for (int w_i = 0; w_i < w; w_i++){
            if (seg_idx_ptr[h_i * w + w_i] != 0 && seg_idx_ptr[h_i * w + w_i] != 1){
                valid_ri.push_back(ri_ptr[h_i * w + w_i]);
                valid_idx.push_back(w_i);
            }
        }
        int valid_length = valid_idx.size();

        if (valid_length < segments + feature_region * 2 + 1)
            continue;

        vector<float> valid_feat(valid_length, 0);
        vector<pair<float, int> > valid_feat_map;
        for (int s_i = feature_region; s_i < valid_length - feature_region; s_i++){
            for (int k = -feature_region; k <= feature_region; k++){
                valid_feat[s_i] += valid_ri[s_i + k] - valid_ri[s_i];
            }
            valid_feat[s_i] = valid_feat[s_i] * valid_feat[s_i];
            valid_feat[s_i] /= 2 * feature_region;
            valid_feat[s_i] /= valid_ri[s_i];
            feat_ptr[h_i * w + valid_idx[s_i]] = valid_feat[s_i];
            valid_feat_map.push_back(make_pair(valid_feat[s_i], s_i));
        }

        for (int j = 0; j < segments; j++){
            int sp = floor(valid_feat_map.size() / segments) * j;
            int ep = floor(valid_feat_map.size() / segments) * (j + 1);

            int largest_picked_num = 0;
            sort(valid_feat_map.begin() + sp, valid_feat_map.begin() + ep);
            for (int i = ep - 1; i >= sp; i--){
                int idx = valid_feat_map[i].second;
                valid_feat_map[i].first = 0;
                if (cloud_neighbors_picked[h_i][valid_idx[idx]] == 0)
                    if (mark_as_picked(ri_ptr, cloud_neighbors_picked, h_i, valid_idx[idx], feature_region)){
                        largest_picked_num += 1;
                        if (largest_picked_num < sharp_num)
                            kp_ptr[h_i * w + valid_idx[idx]] = 3;
                        else if (largest_picked_num < less_sharp_num)
                            kp_ptr[h_i * w + valid_idx[idx]] = 2;
                        else
                            break;
                    }

            }

            int smallest_picked_num = 0;
            sort(valid_feat_map.begin() + sp, valid_feat_map.begin() + ep);
            for (int i = sp; i < ep; i++){
                if (valid_feat_map[i].first == 0)
                    continue;
                int idx = valid_feat_map[i].second;
                valid_feat_map[i].first = 0;
                if (cloud_neighbors_picked[h_i][valid_idx[idx]] == 0)
                    if (mark_as_picked(ri_ptr, cloud_neighbors_picked, h_i, valid_idx[idx], feature_region)){
                        smallest_picked_num += 1;
                        if (smallest_picked_num < flat_num)
                            kp_ptr[h_i * w + valid_idx[idx]] = 1;
                        else
                            break;
                    }
            }
        }
    }

  // reshape array to match input shape
    sharp_feat_map.resize({h, w});
    key_point_map.resize({h, w});

    return std::make_tuple(sharp_feat_map, key_point_map);
}



std::tuple<py::array_t<float>, py::array_t<int>> extract_features(py::array_t<float> range_image, int feature_region, int segments, int sharp_num, int less_sharp_num, int flat_num) {
    // input array
    py::buffer_info ri_buf = range_image.request();
    float *ri_ptr = (float *) ri_buf.ptr;
    int h = ri_buf.shape[0];
    int w = ri_buf.shape[1];

    // return array
    py::array_t<float> sharp_feat_map = py::array_t<float>(ri_buf.size);
    py::array_t<int> key_point_map = py::array_t<int>(ri_buf.size);
    py::buffer_info feat_buf = sharp_feat_map.request();
    py::buffer_info kp_buf = key_point_map.request();
    float *feat_ptr = (float *) feat_buf.ptr;
    int *kp_ptr = (int *) kp_buf.ptr;

    // key_point_map initialization
    for (int h_i = 0; h_i < h; h_i++) {
        for (int w_i = 0; w_i < w; w_i++){
            kp_ptr[h_i * w + w_i] = 0;
        }
    }

    vector<vector<int>> cloud_neighbors_picked(h, vector<int>(w, 0));

    for (int h_i = 0; h_i < h; h_i++) {
        vector<float> valid_ri;
        vector<int> valid_idx;

        for (int w_i = 0; w_i < w; w_i++){
            if (ri_ptr[h_i * w + w_i] != 0){
                valid_ri.push_back(ri_ptr[h_i * w + w_i]);
                valid_idx.push_back(w_i);
            }
        }
        int valid_length = valid_idx.size();

        if (valid_length < segments + feature_region * 2 + 1)
            continue;

        vector<float> valid_feat(valid_length, 0);
        vector<pair<float, int> > valid_feat_map;
        for (int s_i = feature_region; s_i < valid_length - feature_region; s_i++){
            for (int k = -feature_region; k <= feature_region; k++){
                valid_feat[s_i] += valid_ri[s_i + k] - valid_ri[s_i];
            }
            valid_feat[s_i] = valid_feat[s_i] * valid_feat[s_i];
            valid_feat[s_i] /= 2 * feature_region;
            valid_feat[s_i] /= valid_ri[s_i];
            feat_ptr[h_i * w + valid_idx[s_i]] = valid_feat[s_i];
            valid_feat_map.push_back(make_pair(valid_feat[s_i], s_i));
        }

        for (int j = 0; j < segments; j++){
            int sp = floor(valid_feat_map.size() / segments) * j;
            int ep = floor(valid_feat_map.size() / segments) * (j + 1);

            int largest_picked_num = 0;
            sort(valid_feat_map.begin() + sp, valid_feat_map.begin() + ep);
            for (int i = ep - 1; i >= sp; i--){
                int idx = valid_feat_map[i].second;
                valid_feat_map[i].first = 0;
                if (cloud_neighbors_picked[h_i][valid_idx[idx]] == 0)
                    if (mark_as_picked(ri_ptr, cloud_neighbors_picked, h_i, valid_idx[idx], feature_region)){
                        largest_picked_num += 1;
                        if (largest_picked_num < sharp_num)
                            kp_ptr[h_i * w + valid_idx[idx]] = 3;
                        else if (largest_picked_num < less_sharp_num)
                            kp_ptr[h_i * w + valid_idx[idx]] = 2;
                        else
                            break;
                    }

            }

            int smallest_picked_num = 0;
            sort(valid_feat_map.begin() + sp, valid_feat_map.begin() + ep);
            for (int i = sp; i < ep; i++){
                if (valid_feat_map[i].first == 0)
                    continue;
                int idx = valid_feat_map[i].second;
                valid_feat_map[i].first = 0;
                if (cloud_neighbors_picked[h_i][valid_idx[idx]] == 0)
                    if (mark_as_picked(ri_ptr, cloud_neighbors_picked, h_i, valid_idx[idx], feature_region)){
                        smallest_picked_num += 1;
                        if (smallest_picked_num < flat_num)
                            kp_ptr[h_i * w + valid_idx[idx]] = 1;
                        else
                            break;
                    }
            }
        }
    }

  // reshape array to match input shape
    sharp_feat_map.resize({h, w, 1});
    key_point_map.resize({h, w, 1});

    return std::make_tuple(sharp_feat_map, key_point_map);
}


py::array_t<int> segment_index_clean(py::array_t<int> seg_idx) {
    // input array
    py::buffer_info si_buf = seg_idx.request();
    int *si_ptr = (int *) si_buf.ptr;
    int h = si_buf.shape[0];
    int w = si_buf.shape[1];

    int cur, next, next2;
    for (int h_i = 0; h_i < h; h_i++) {
        for (int w_i = 0; w_i < w - 2; w_i++){
            cur = si_ptr[h_i * w + w_i];
            next = si_ptr[h_i * w + w_i + 1];
            next2 = si_ptr[h_i * w + w_i + 2];
            if (next2 == cur && next != cur){
                si_ptr[h_i * w + w_i + 1] = cur;
            }
        }
    }

    return seg_idx;
}

py::array_t<float> intra_predict(py::array_t<int> seg_idx, py::array_t<float> model_param, py::array_t<float> transform_map) {
    // input array
    py::buffer_info si_buf = seg_idx.request();
    int *si_ptr = (int *) si_buf.ptr;
    py::buffer_info mp_buf = model_param.request();
    float *mp_ptr = (float *) mp_buf.ptr;
    py::buffer_info tm_buf = transform_map.request();
    float *tm_ptr = (float *) tm_buf.ptr;
    int h = si_buf.shape[0];
    int w = si_buf.shape[1];

    // return array
    auto pred_range_image = py::array_t<float>(si_buf.size);
    py::buffer_info pred_buf = pred_range_image.request();
    float *pred_ptr = (float *) pred_buf.ptr;

    for (int h_i = 0; h_i < h; h_i++) {
        for (int w_i = 0; w_i < w; w_i++){
            int model_idx = si_ptr[h_i * w + w_i];
            float param[4] = {mp_ptr[model_idx * 4],
                                       mp_ptr[model_idx * 4 + 1],
                                       mp_ptr[model_idx * 4 + 2],
                                       mp_ptr[model_idx * 4 + 3]};
            if (param[0] + param[1] + param[2] == 0){
                pred_ptr[h_i * w + w_i] = param[3];
            }
            else{
                float r_plane = -param[3] / (param[0] * tm_ptr[(h_i * w + w_i) * 3] +
                                             param[1] * tm_ptr[(h_i * w + w_i) * 3 + 1] +
                                             param[2] * tm_ptr[(h_i * w + w_i) * 3 + 2]);
                pred_ptr[h_i * w + w_i] = r_plane;
            }
        }
    }

    pred_range_image.resize({h, w, 1});
    return pred_range_image;
}


py::array_t<int> uniform_quantize(py::array_t<int> seg_idx, py::array_t<float> residual, float acc) {
    // input array
    py::buffer_info si_buf = seg_idx.request();
    int *si_ptr = (int *) si_buf.ptr;
    py::buffer_info ri_buf = residual.request();
    float *ri_ptr = (float *) ri_buf.ptr;
    int h = si_buf.shape[0];
    int w = si_buf.shape[1];

    int cluster_num = 0;
    for (int h_i = 0; h_i < h; h_i++)
        for (int w_i = 0; w_i < w; w_i++)
            if (si_ptr[h_i * w + w_i] > cluster_num)
                cluster_num = si_ptr[h_i * w + w_i];
    cluster_num += 1;

    vector< vector<int> > quantized_residual;
    for (int i = 0; i < cluster_num; i++){
        vector<int> cur_qr;
        quantized_residual.push_back(cur_qr);
    }

    int total_num = 0;
    for (int h_i = 0; h_i < h; h_i++) {
        for (int w_i = 0; w_i < w; w_i++){
            int idx = si_ptr[h_i * w + w_i];
            if (idx != 1){
                quantized_residual[idx].push_back(round(ri_ptr[h_i * w + w_i] / acc));
                total_num++;
            }
        }
    }

    // return array
    auto quantized_residual_collect = py::array_t<int>(total_num);
    py::buffer_info qr_buf = quantized_residual_collect.request();
    int *qr_ptr = (int *) qr_buf.ptr;

    int k = 0;
    for (int i = 0; i < cluster_num; i++)
        for (std::vector<int>::iterator it = quantized_residual[i].begin() ; it != quantized_residual[i].end(); it++){
            qr_ptr[k] = *it;
            k++;
        }

    return quantized_residual_collect;
}


std::tuple<py::array_t<int>, py::array_t<int>> nonuniform_quantize(py::array_t<int> seg_idx,
                                     py::array_t<float> residual, py::array_t<int> key_point_map,
                                     py::array_t<int> level_kp_num, py::array_t<float> level_acc, int ground_level) {
    // input array
    py::buffer_info si_buf = seg_idx.request();
    int *si_ptr = (int *) si_buf.ptr;
    py::buffer_info ri_buf = residual.request();
    float *ri_ptr = (float *) ri_buf.ptr;
    py::buffer_info kp_buf = key_point_map.request();
    int *kp_ptr = (int *) kp_buf.ptr;
    py::buffer_info lkpn_buf = level_kp_num.request();
    int *lkpn_ptr = (int *) lkpn_buf.ptr;
    py::buffer_info la_buf = level_acc.request();
    float *la_ptr = (float *) la_buf.ptr;
    int h = si_buf.shape[0];
    int w = si_buf.shape[1];
    int level_num = la_buf.shape[0];

    int cluster_num = 0;
    for (int h_i = 0; h_i < h; h_i++)
        for (int w_i = 0; w_i < w; w_i++)
            if (si_ptr[h_i * w + w_i] > cluster_num)
                cluster_num = si_ptr[h_i * w + w_i];
    cluster_num += 1;

    vector< vector<float> > quantized_residual;
    vector<int> kp_num;
    vector<int> p_num;
    vector<int> salience_level;
    vector<float> cluster_acc;
    for (int i = 0; i < cluster_num; i++){
        vector<float> cur_qr;
        quantized_residual.push_back(cur_qr);
        kp_num.push_back(0);
        p_num.push_back(0);
        salience_level.push_back(0);
        cluster_acc.push_back(0);
    }

    int total_num = 0;
    for (int h_i = 0; h_i < h; h_i++) {
        for (int w_i = 0; w_i < w; w_i++){
            int idx = si_ptr[h_i * w + w_i];
            if (idx == 1) continue;
            if (kp_ptr[h_i * w + w_i] > 0)
                kp_num[idx] += 1;
            p_num[idx] += 1;
            quantized_residual[idx].push_back(ri_ptr[h_i * w + w_i]);
            total_num += 1;
        }
    }
    for (int i = 0; i < cluster_num; i++){
        if (i == 0)
            salience_level[0] = ground_level;  // ground points
        else if (i == 1)
            salience_level[1] = level_num - 1;  // zero points;
        else{
            if (p_num[i] < 30)
                salience_level[i] = level_num - 1;
            else
                for (int l = 0; l < level_num; l++){
                    if (kp_num[i] >= lkpn_ptr[l]){
                        salience_level[i] = l;
                        break;
                    }
                }
        }
        cluster_acc[i] = la_ptr[salience_level[i]];
    }

    // return array
    auto quantized_residual_collect = py::array_t<int>(total_num);
    py::buffer_info qr_buf = quantized_residual_collect.request();
    int *qr_ptr = (int *) qr_buf.ptr;
    auto salience_level_array = py::array_t<int>(cluster_num);
    py::buffer_info sl_buf = salience_level_array.request();
    int *sl_ptr = (int *) sl_buf.ptr;

    int k = 0;
    for (int i = 0; i < cluster_num; i++){
        sl_ptr[i] = salience_level[i];
        for (std::vector<float>::iterator it = quantized_residual[i].begin() ; it != quantized_residual[i].end(); it++){
            qr_ptr[k] = round(*it / cluster_acc[i]);
            k++;
        }
    }
    return std::make_tuple(quantized_residual_collect, salience_level_array);
}


py::array_t<float> point_cloud_to_range_image_even(py::array_t<float> point_cloud, int H, int W, float horizontal_FOV,
float vertical_max, float vertical_min) {
    // input array
    py::buffer_info pc_buf = point_cloud.request();
    float *pc_ptr = (float *) pc_buf.ptr;
    int N = pc_buf.shape[0];

    // return array
    auto range_image = py::array_t<float>(H * W);
    py::buffer_info ri_buf = range_image.request();
    float *ri_ptr = (float *) ri_buf.ptr;

    for (int i = 0; i < H * W; i++)
        ri_ptr[i] = 0;

    for (int i = 0; i < N; i++){
        float x = pc_ptr[i * 3];
        float y = pc_ptr[i * 3 + 1];
        float z = pc_ptr[i * 3 + 2];
        float depth = sqrt(x * x + y * y + z * z);
        float horizontal_angle = atan2(y, x);
        if (horizontal_angle < 0)
            horizontal_angle += 2 * 3.14159265;
        float vertical_angle = atan2(z, sqrt(x*x + y*y));
        int col = round(horizontal_angle / horizontal_FOV * W);
        col = col % W;
        float vertical_resolution = (vertical_max - vertical_min) / (H - 1);
        int row = round((vertical_angle - vertical_min) / vertical_resolution);
        if (row >= H)
            row = H - 1;
        if (row < 0)
            row = 0;
        if (ri_ptr[row * W + col] == 0 || depth < ri_ptr[row * W + col])
            ri_ptr[row * W + col] = depth;

    }
//    cout << "N: " << N << endl;

    range_image.resize({H, W});
    return range_image;
}



py::array_t<float> point_modeling(py::array_t<float> range_image, py::array_t<int> seg_idx) {
    // input array
    py::buffer_info ri_buf = range_image.request();
    float *ri_ptr = (float *) ri_buf.ptr;
    py::buffer_info si_buf = seg_idx.request();
    int *si_ptr = (int *) si_buf.ptr;
    int h = si_buf.shape[0];
    int w = si_buf.shape[1];


    int cluster_num = 0;
    for (int h_i = 0; h_i < h; h_i++)
        for (int w_i = 0; w_i < w; w_i++)
            if (si_ptr[h_i * w + w_i] > cluster_num)
                cluster_num = si_ptr[h_i * w + w_i];
    cluster_num += 1;

    vector< vector<float> > cluster_points;
    for (int i = 0; i < cluster_num; i++){
        vector<float> cur_cluster;
        cluster_points.push_back(cur_cluster);
    }

    for (int h_i = 0; h_i < h; h_i++) {
        for (int w_i = 0; w_i < w; w_i++){
            int idx = si_ptr[h_i * w + w_i];
            if (idx != 0 && idx != 1){
                cluster_points[idx].push_back(ri_ptr[h_i * w + w_i]);
            }
        }
    }


    // return array
    auto point_model = py::array_t<float>(cluster_num);
    py::buffer_info qm_buf = point_model.request();
    float *qm_ptr = (float *) qm_buf.ptr;


    for (int i = 0; i < cluster_num; i++){
        if (i == 0 || i == 1)
            qm_ptr[i] = 0.0;
        else
            qm_ptr[i] = accumulate( cluster_points[i].begin(), cluster_points[i].end(), 0.0)/cluster_points[i].size();
    }

    return point_model;
}


std::tuple<py::array_t<int>, py::array_t<int>> extract_contour(py::array_t<int> idx_map) {
    // input array
    py::buffer_info im_buf = idx_map.request();
    int *im_ptr = (int *) im_buf.ptr;
    int h = im_buf.shape[0];
    int w = im_buf.shape[1];

    // return array
    auto contour_map = py::array_t<int>(h * w);
    py::buffer_info cm_buf = contour_map.request();
    int *cm_ptr = (int *) cm_buf.ptr;

    vector<int> idx_sequence_vec;
    for (int h_i = 0; h_i < h; h_i++){
        idx_sequence_vec.push_back(im_ptr[h_i * w]);
        cm_ptr[h_i * w] = 1;
        for (int w_i = 1; w_i < w; w_i++)
            if (im_ptr[h_i * w + w_i] - im_ptr[h_i * w + w_i - 1] != 0){
                idx_sequence_vec.push_back(im_ptr[h_i * w + w_i]);
                cm_ptr[h_i * w + w_i] = 1;
            }
            else{
                cm_ptr[h_i * w + w_i] = 0;
            }
    }

    // return array
    int seq_size = idx_sequence_vec.size();
    auto idx_sequence = py::array_t<int>(seq_size);
    py::buffer_info is_buf = idx_sequence.request();
    int *is_ptr = (int *) is_buf.ptr;
    for (int i = 0; i < seq_size; i++){
        is_ptr[i] = idx_sequence_vec[i];
    }

    contour_map.resize({h, w});
    return std::make_tuple(contour_map, idx_sequence);
}


py::array_t<int> recover_map(py::array_t<int> contour_map, py::array_t<int> idx_sequence) {
    // input array
    py::buffer_info cm_buf = contour_map.request();
    int *cm_ptr = (int *) cm_buf.ptr;
    py::buffer_info is_buf = idx_sequence.request();
    int *is_ptr = (int *) is_buf.ptr;
    int h = cm_buf.shape[0];
    int w = cm_buf.shape[1];
    int l = is_buf.shape[0];

    // return array
    auto idx_map = py::array_t<int>(h * w);
    py::buffer_info im_buf = idx_map.request();
    int *im_ptr = (int *) im_buf.ptr;
    
    int pointer = 0;
    for (int i = 0; i < l; i++){
        int index = is_ptr[i];
        im_ptr[pointer] = index;
        pointer++;
        if (pointer >= h * w)
            break;
        while (cm_ptr[pointer] == 0){
            im_ptr[pointer] = index;
            pointer++;
            if (pointer >= h * w)
                break;
        }
    }

    idx_map.resize({h, w});
    return idx_map;
}


// feature extractor
PYBIND11_MODULE(feature_extractor_cpp, m) {
        m.doc() = "Find point's feature and search key points using pybind11"; // optional module docstring

        m.def("extract_features_with_segment", &extract_features_with_segment, "Find point's feature and search key points after FPS segmentation.");
        m.def("extract_features", &extract_features, "Find point's feature and search key points for whole range image.");

        m.def("segment_index_clean", &segment_index_clean, "Clean the index map.");
}

// segmentation modules
PYBIND11_MODULE(segment_utils_cpp, m) {
    m.doc() = "Cpp modules of segment_utils.py using pybind11"; // optional module docstring

    m.def("intra_predict", &intra_predict, "Intra-prediction with segmentation and modeling information.");
    m.def("point_modeling", &point_modeling, "Point modeling module for each cluster.");

}

// quantization modules
PYBIND11_MODULE(quantization_utils_cpp, m) {
    m.doc() = "Cpp modules of class QuantizationModule in compress_utils.py using pybind11";

    m.def("uniform_quantize", &uniform_quantize, "Uniform quantization (without zero points).");
    m.def("nonuniform_quantize", &nonuniform_quantize, "Non-uniform quantization (without zero points).");
}

// contour extractor modules
PYBIND11_MODULE(contour_utils_cpp, m) {
    m.doc() = "Cpp modules of contour extractor in contour_utils.py using pybind11";

    m.def("extract_contour", &extract_contour, "Extract contour and index sequence from segmentation map.");
    m.def("recover_map", &recover_map, "Recover segmentation map from contour and index sequence.");
}

// dataset transformer modules
PYBIND11_MODULE(dataset_utils_cpp, m) {
    m.doc() = "Cpp modules of transformer.py using pybind11"; // optional module docstring

    m.def("point_cloud_to_range_image_even", &point_cloud_to_range_image_even, "Project point cloud into range image with even vertical channels distribution.");
}





