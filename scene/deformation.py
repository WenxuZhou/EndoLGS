import os
import torch
import torch.nn as nn
import torch.nn.init as init
from scene.hexplane import HexPlaneField

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform, self.discrete_coff_generator, self.lang_deform = self.create_net()

    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        language_feature_hiddendim = int(os.getenv("language_feature_hiddendim", 3))
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, int(os.getenv("centers_num",3)))),\
            nn.Sequential(nn.ReLU(), nn.Linear(self.args.timebase_pe * 2 + 1 + language_feature_hiddendim, self.W),
                              nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(),
                              nn.Linear(self.W, language_feature_hiddendim))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature
        h = self.feature_out(h)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None, lang_emb=None, init_centers=False):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb, lang_emb, init_centers)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb, lang_emb, init_centers=False):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()

        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = rays_pts_emb[:, :3] + dx

        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds

        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr

        if self.args.no_do:
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(hidden)
            opacity = opacity_emb[:,:1] + do

        if os.getenv('use_discrete_lang_f', 'f') == 't' and init_centers == False:
            lang_feature = lang_emb[:,
                           :int(os.getenv("language_feature_hiddendim", 3) * int(os.getenv("centers_num", 3)))]
            lang_feature = lang_feature.view(lang_feature.shape[0], int(os.getenv("centers_num", 3)), -1)
            lang_feature = lang_feature / (torch.norm(lang_feature, dim=-1, keepdim=True))
            coff = self.discrete_coff_generator(hidden)

            lang_feature = torch.matmul(coff.unsqueeze(1), lang_feature).squeeze(1)
            lang_feature = lang_feature / (torch.norm(lang_feature, dim=1, keepdim=True) + + 1e-9)
        else:
            coff = None
            assert (init_centers and self.args.no_dlang) == False, " Dlang must be enabled when initialized centers"
            language_dim = int(os.getenv("language_feature_hiddendim", 3))

            if self.args.no_dlang:
                lang_feature = lang_emb[:, :language_dim]
            else:
                if os.getenv("use_tribute_dlang", "f") == "t":
                    dlang = self.lang_deform(hidden)
                else:
                    dlang = self.lang_deform(torch.concat((lang_emb, time_emb), dim=1))
                if os.getenv("no_resnet", 'f') == 't':
                 lang_feature = dlang
                else:
                    lang_feature = lang_emb[:, :language_dim] + dlang
                lang_feature = lang_feature / (lang_feature.norm(dim=-1, keepdim=True) + 1e-9)

        return pts, scales, rotations, opacity, lang_feature, coff

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        return list(self.grid.parameters())

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe

        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))

        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None, lang=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel, lang)
        else:
            return self.forward_static(point)

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None, lang=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        means3D, scales, rotations, opacity, lang, coff = self.deformation_net(point, scales, rotations, opacity, times_sel, lang)

        return means3D, scales, rotations, opacity, lang, coff
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
