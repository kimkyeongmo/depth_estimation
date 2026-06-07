import torch
from gaussian_renderer import render

# 네트워크 제한 크기와 완벽하게 일치하는 필터링 상한선
SCALE_MAX = 0.01
ZNEAR     = 0.01

def pts2render(data, bg_color, is_train=True):
    bs = data['lmain']['img'].shape[0]
    render_novel_list = []

    for i in range(bs):
        # 리스트 초기화 수정
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []

        for view in ['lmain']:
            valid_i   = data[view]['pts_valid'][i, :]
            xyz_i     = data[view]['xyz'][i, :, :]
            rgb_i     = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot_i     = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i   = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)

        if pts_xyz_i.shape[0] == 0:
            device    = data['lmain']['img'].device
            pts_xyz_i = torch.zeros((1, 3), device=device)
            pts_rgb_i = torch.zeros((1, 3), device=device)
            rot_i     = torch.zeros((1, 4), device=device); rot_i[:, 0] = 1.0
            scale_i   = torch.ones((1, 3),  device=device) * 1e-5
            opacity_i = torch.zeros((1, 1), device=device)

        else:
            pts_rgb_i = torch.concat(rgb_i_valid,   dim=0)
            rot_i     = torch.concat(rot_i_valid,   dim=0)
            scale_i   = torch.concat(scale_i_valid, dim=0)
            opacity_i = torch.concat(opacity_i_valid, dim=0)

            defense_mask = (
                torch.isfinite(pts_xyz_i).all(dim=-1) &
                torch.isfinite(pts_rgb_i).all(dim=-1) &
                torch.isfinite(rot_i).all(dim=-1)     &
                torch.isfinite(scale_i).all(dim=-1)   &
                torch.isfinite(opacity_i.squeeze(-1))
            )

            # 비정상 노이즈 가우시안 폐기
            defense_mask = defense_mask & (scale_i.max(dim=-1)[0] <= SCALE_MAX)

            if 'novel_view' in data and 'world_view_transform' in data['novel_view']:
                vm      = data['novel_view']['world_view_transform'][i]
                xyz_cam = torch.matmul(pts_xyz_i, vm[:3, :3]) + vm[3, :3]
                z_vals  = xyz_cam[..., 2]
                defense_mask = defense_mask & (z_vals > ZNEAR)

            pts_xyz_i = pts_xyz_i[defense_mask]
            pts_rgb_i = pts_rgb_i[defense_mask]
            rot_i     = rot_i[defense_mask]
            scale_i   = scale_i[defense_mask]
            opacity_i = opacity_i[defense_mask]

            if pts_xyz_i.shape[0] == 0:
                device    = data['lmain']['img'].device
                pts_xyz_i = torch.zeros((1, 3), device=device)
                pts_rgb_i = torch.zeros((1, 3), device=device)
                rot_i     = torch.zeros((1, 4), device=device); rot_i[:, 0] = 1.0
                scale_i   = torch.ones((1, 3),  device=device) * 1e-5
                opacity_i = torch.zeros((1, 1), device=device)

        render_pkg = render(
            data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i,
            bg_color=bg_color
        )
        
        if isinstance(render_pkg, dict):
            render_novel_list.append(render_pkg["render"].unsqueeze(0))
        elif isinstance(render_pkg, tuple):
            render_novel_list.append(render_pkg[0].unsqueeze(0))
        else:
            render_novel_list.append(render_pkg.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    
    return data