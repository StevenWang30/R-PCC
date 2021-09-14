//c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) feature_extraction.cpp -o feature_extractor$(python3-config --extension-suffix)
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



//        int s = feature_region
//        int e = pc_h.shape[0] - feature_region - 1
        for (int j = 0; j < segments; j++){
//            sp = s + floor((e - s) / segments) * j
//            ep = s + floor((e - s) / segments) * (j + 1)
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
//  return sharp_feat_map;
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



//        int s = feature_region
//        int e = pc_h.shape[0] - feature_region - 1
        for (int j = 0; j < segments; j++){
//            sp = s + floor((e - s) / segments) * j
//            ep = s + floor((e - s) / segments) * (j + 1)
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
//  return sharp_feat_map;
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
//
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


    vector<int> quantized_residual_sort;
    int k = 0;
    for (int i = 0; i < cluster_num; i++)
        for (std::vector<int>::iterator it = quantized_residual[i].begin() ; it != quantized_residual[i].end(); it++){
            qr_ptr[k] = *it;
            k++;
        }


    return quantized_residual_collect;
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

// feature extractor
PYBIND11_MODULE(feature_extractor_cpp, m) {
        m.doc() = "Find point's feature and search key points using pybind11"; // optional module docstring

        m.def("extract_features_with_segment", &extract_features_with_segment, "Find point's feature and search key points after FPS segmentation.");
        m.def("extract_features", &extract_features, "Find point's feature and search key points for whole range image");

        m.def("segment_index_clean", &segment_index_clean, "Find point's feature and search key points for whole range image");
//        m.def("nonuniform_quantize", &nonuniform_quantize, "Find point's feature and search key points.");
}

// segmentation modules
PYBIND11_MODULE(segment_utils_cpp, m) {
    m.doc() = "cpp modules of segment_utils.py using pybind11"; // optional module docstring

    m.def("intra_predict", &intra_predict, "intra-prediction with segmentation and modeling information.");
    m.def("point_modeling", &point_modeling, "intra-prediction with segmentation and modeling information.");

}

// quantization modules
PYBIND11_MODULE(quantization_utils_cpp, m) {
    m.doc() = "cpp modules of segment_utils.py using pybind11"; // optional module docstring

    m.def("uniform_quantize", &uniform_quantize, "intra-prediction with segmentation and modeling information.");
}


// dataset transformer modules
PYBIND11_MODULE(dataset_utils_cpp, m) {
    m.doc() = "cpp modules of transformer.py using pybind11"; // optional module docstring

    m.def("point_cloud_to_range_image_even", &point_cloud_to_range_image_even, "intra-prediction with segmentation and modeling information.");
}





