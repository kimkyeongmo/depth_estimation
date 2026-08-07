import open3d as o3d
import numpy as np
import time
import glob
import os

ply_dir = "outputs_120cm3/120cm_1k_ply"
ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))

if not ply_files:
    print("PLY 파일을 찾을 수 없습니다.")
    exit()

vis = o3d.visualization.Visualizer()
# 창 생성 시도 및 결과 확인
success = vis.create_window(window_name="PLY Sequence Viewer", width=1280, height=720, visible=True)

if not success:
    print("Open3D 창 생성 실패. OpenGL 드라이버나 디스플레이 설정을 확인하십시오.")
    exit()

# 창이 생성된 후 렌더링 옵션 접근
render_option = vis.get_render_option()
if render_option:
    render_option.point_size = 1.0
    render_option.background_color = np.asarray([0, 0, 0])

flip_transform = np.array([[1,  0,  0, 0],
                           [0, -1,  0, 0],
                           [0,  0, -1, 0],
                           [0,  0,  0, 1]])

pcd = o3d.io.read_point_cloud(ply_files[0])
pcd.transform(flip_transform)
vis.add_geometry(pcd)

for i in range(1, len(ply_files)):
    new_pcd = o3d.io.read_point_cloud(ply_files[i])
    new_pcd.transform(flip_transform)
    
    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors
    
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


vis.destroy_window()