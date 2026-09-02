#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__global__ void ray_scoring_cuda_kernel(
    const float* __restrict__ cam_positions,     // (M, 3)
    const float* __restrict__ unseen_points,     // (N, 3)
    const float* __restrict__ unseen_normals,    // (N, 3)
    const float* __restrict__ triangles,         // (T, 3, 3) = T * 9 floats
    int M,
    int N,
    int T,
    float backface_margin,
    float occlusion_epsilon_m,
    uint8_t* __restrict__ visibility_mask,       // (M, N)
    int32_t* __restrict__ scores                 // (M,)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = M * N;
    if (idx >= total_threads) return;

    int m = idx / N;
    int n = idx % N;

    // Camera and target point coordinates
    float3 cam = make_float3(cam_positions[m * 3 + 0], cam_positions[m * 3 + 1], cam_positions[m * 3 + 2]);
    float3 pt  = make_float3(unseen_points[n * 3 + 0], unseen_points[n * 3 + 1], unseen_points[n * 3 + 2]);
    float3 nrm = make_float3(unseen_normals[n * 3 + 0], unseen_normals[n * 3 + 1], unseen_normals[n * 3 + 2]);

    // Vector from point to camera
    float3 to_cam = make_float3(cam.x - pt.x, cam.y - pt.y, cam.z - pt.z);
    float dist_sq = to_cam.x * to_cam.x + to_cam.y * to_cam.y + to_cam.z * to_cam.z;
    float dist = sqrtf(fmaxf(dist_sq, 1e-12f));

    // 1. Front-facing check
    float cos_angle = (to_cam.x * nrm.x + to_cam.y * nrm.y + to_cam.z * nrm.z) / dist;
    if (cos_angle <= backface_margin) {
        visibility_mask[idx] = 0;
        return;
    }

    // 2. Occlusion test: ray from camera toward target point
    float3 ray_orig = cam;
    float3 ray_dir = make_float3(-to_cam.x, -to_cam.y, -to_cam.z);
    float t_target = 1.0f - occlusion_epsilon_m / fmaxf(dist, 1e-6f);
    bool occluded = false;

    // Moller-Trumbore ray-triangle intersection
    for (int t = 0; t < T; ++t) {
        const float* tri_ptr = &triangles[t * 9];
        float3 v0 = make_float3(tri_ptr[0], tri_ptr[1], tri_ptr[2]);
        float3 v1 = make_float3(tri_ptr[3], tri_ptr[4], tri_ptr[5]);
        float3 v2 = make_float3(tri_ptr[6], tri_ptr[7], tri_ptr[8]);

        float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

        // h = cross(ray_dir, e2)
        float3 h = make_float3(
            ray_dir.y * e2.z - ray_dir.z * e2.y,
            ray_dir.z * e2.x - ray_dir.x * e2.z,
            ray_dir.x * e2.y - ray_dir.y * e2.x
        );

        float a = e1.x * h.x + e1.y * h.y + e1.z * h.z;
        if (fabsf(a) < 1e-9f) continue;

        float f = 1.0f / a;
        float3 s = make_float3(ray_orig.x - v0.x, ray_orig.y - v0.y, ray_orig.z - v0.z);
        float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
        if (u < 0.0f || u > 1.0f) continue;

        // q = cross(s, e1)
        float3 q = make_float3(
            s.y * e1.z - s.z * e1.y,
            s.z * e1.x - s.x * e1.z,
            s.x * e1.y - s.y * e1.x
        );

        float v = f * (ray_dir.x * q.x + ray_dir.y * q.y + ray_dir.z * q.z);
        if (v < 0.0f || (u + v) > 1.0f) continue;

        float t_hit = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);
        if (t_hit > 1e-5f && t_hit < t_target) {
            occluded = true;
            break;
        }
    }

    uint8_t vis = occluded ? 0 : 1;
    visibility_mask[idx] = vis;
    if (vis) {
        atomicAdd(&scores[m], 1);
    }
}

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
) {
    int total = M * N;
    if (total == 0) return;

    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    ray_scoring_cuda_kernel<<<blocks, threads_per_block, 0, stream>>>(
        cam_positions,
        unseen_points,
        unseen_normals,
        triangles,
        M,
        N,
        T,
        backface_margin,
        occlusion_epsilon_m,
        visibility_mask,
        scores
    );
}
