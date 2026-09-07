conda run -n rob_env python -c "
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('captures/full_pipeline/coverage_visualization.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
"