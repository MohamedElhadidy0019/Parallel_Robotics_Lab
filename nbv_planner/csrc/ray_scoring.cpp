#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void launch_ray_scoring_kernel(
    const float* cam_positions,
    const float* unseen_points,
    const float* unseen_normals,
    const float* triangles,
    int M,
    int N,
    int T,
    float backface_margin,
    float occlusion_epsilon_m,
    uint8_t* visibility_mask,
    int32_t* scores,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> score_candidate_views_cuda(
    torch::Tensor cam_positions,     // (M, 3) float32
    torch::Tensor unseen_points,     // (N, 3) float32
    torch::Tensor unseen_normals,    // (N, 3) float32
    torch::Tensor triangles,         // (T, 3, 3) float32
    float backface_margin,
    float occlusion_epsilon_m
) {
    TORCH_CHECK(cam_positions.is_cuda(), "cam_positions must be a CUDA tensor");
    TORCH_CHECK(unseen_points.is_cuda(), "unseen_points must be a CUDA tensor");
    TORCH_CHECK(unseen_normals.is_cuda(), "unseen_normals must be a CUDA tensor");
    TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");

    TORCH_CHECK(cam_positions.dtype() == torch::kFloat32, "cam_positions must be float32");
    TORCH_CHECK(unseen_points.dtype() == torch::kFloat32, "unseen_points must be float32");
    TORCH_CHECK(unseen_normals.dtype() == torch::kFloat32, "unseen_normals must be float32");
    TORCH_CHECK(triangles.dtype() == torch::kFloat32, "triangles must be float32");

    cam_positions = cam_positions.contiguous();
    unseen_points = unseen_points.contiguous();
    unseen_normals = unseen_normals.contiguous();
    triangles = triangles.contiguous();

    int M = cam_positions.size(0);
    int N = unseen_points.size(0);
    int T = triangles.size(0);

    auto options_uint8 = torch::TensorOptions().dtype(torch::kUInt8).device(cam_positions.device());
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(cam_positions.device());

    torch::Tensor visibility_mask = torch::zeros({M, N}, options_uint8);
    torch::Tensor scores = torch::zeros({M}, options_int32);

    if (M == 0 || N == 0) {
        return std::make_tuple(visibility_mask, scores);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_ray_scoring_kernel(
        cam_positions.data_ptr<float>(),
        unseen_points.data_ptr<float>(),
        unseen_normals.data_ptr<float>(),
        triangles.data_ptr<float>(),
        M,
        N,
        T,
        backface_margin,
        occlusion_epsilon_m,
        visibility_mask.data_ptr<uint8_t>(),
        scores.data_ptr<int32_t>(),
        stream
    );

    return std::make_tuple(visibility_mask, scores);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score_candidate_views_cuda", &score_candidate_views_cuda, "Ray-scoring and candidate visibility CUDA kernel");
}
