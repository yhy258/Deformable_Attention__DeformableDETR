import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(h, w, ref):
    new_point = ref.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (w -1)
    new_point[..., 1] = new_point[..., 1] * (h -1)
    return new_point


class DeformableAttention(nn.Module):
    def __init__(self, C, M, K, L, return_attention=False):
        super().__init__()
        self.C = C
        self.M = M
        self.K = K
        self.L = L
        self.C_v = self.C // self.M
        self.return_attention = return_attention
        self.W_prim = nn.Linear(C, C)

        self.z_q = nn.Linear(C, C)
        self.offsets = nn.Linear(C, 2 * M * K * L)

        self.attn_score = nn.Linear(C, M * K * L)

    def forward(self, q, features, ref):
        output = {'attentions': None, 'offsets': None}
        """
            q : 단일 layer의 쿼리
            features : 전체 layers에 대한 input features
            ref : q의 좌표 정보
        """
        BS, H, W, _ = q.size()
        z_q = self.z_q(q)  # [BS, H, W, C]
        offsets = self.offsets(z_q)  # [BS, H, W, 2*M*K*L]
        offsets = offsets.view(BS, H, W, self.M, -1)  # [BS, H, W, M, 2*K*L]

        attn_score = self.attn_score(z_q)  # [BS, H, W, M*K*L]
        attn_score = attn_score.view(BS, H, W, self.M, -1)  # [BS, H, W, M, K*L]
        attn_score = F.softmax(attn_score, dim=-1) # K, L에 대해서 softmax의 합은 1
        if self.return_attention:
            output['attentions'] = attn_score  # # batch, H, W, M, L*K
            output['offsets'] = offsets  # B, H, W, M, 2LK

        offsets = offsets.view(BS, H, W, self.M, self.L, self.K, 2)

        offsets = offsets.permute(0, 3, 4, 5, 1, 2, 6).contiguous()  # [BS, M, L, K, H, W, 2]
        offsets = offsets.view(BS * self.M, self.L, self.K, H, W, 2)  # [BS*M, L, K, H ,W, 2]
        # BS와 M을 한꺼번에. 나중에 쿼리에 대한 좌표인 ref를 이에 맞게 바꿔줄거임.
        # ref의 shape은 BS, H, W, 2 인데, .repeat(M, 1,1,1)을 통해 BS*M, H, W, 2로 만들어줄거임. 이러면 어차피 상관 없음.

        attn_score = attn_score.permute(0, 3, 1, 2, 4).contiguous()  # batch, M, H, W, L*K
        attn_score = attn_score.view(BS * self.M, H * W, -1)  # Batch *M, H*W, LK
        # attn_score도 이에 맞게


        sampled_features_scale_list = []
        for l, feature in enumerate(features):
            w_prim = self.W_prim(feature)  # [BS, H, W ,C]
            bs, h, w, c = w_prim.shape
            phi_ref = phi(h, w, ref)  # 현재 target layer의 h, w에 맞춰서 rescale
            # [bs, h, w, 2]
            phi_ref = phi_ref.repeat(self.M, 1, 1, 1)  # [BS * M , h, w, 2]
            w_prim = w_prim.view(bs, h, w, self.M, self.C_v)  # [bs, h ,w, m, c]

            w_prim = w_prim.permute(0, 3, 4, 1, 2).contiguous()  # bs, M, C ,H,W
            w_prim = w_prim.view(-1, self.C_v, h, w)

            sampled_features = self.compute_sampling(w_prim, phi_ref, offsets, l, h, w)  # compute_sampling 함수

            sampled_features_scale_list.append(sampled_features)  # L * [B*M,k,c,h,w]

            sampled_features_scaled = torch.stack(sampled_features_scale_list, dim=1)
            # B*M, H*W, C_v, LK
            sampled_features_scaled = sampled_features_scaled.permute(0, 4, 5, 3, 1, 2).contiguous()
            sampled_features_scaled = sampled_features_scaled.view(BS * self.M, H * W, self.C_v, -1)
            """
            attention score에 맞게
            Attention_score  [B *M : n, H*W : l, LK : s]
            sampled_features_scaled [B*M : n, H*W : l, C_v : d, LK : s]
            이렇게 shape을 바꾸는 이유는 어차피 L, K에 대해서 전부 sum 해줄거라서. Attention처럼.
            """

            Attention_W_prim_x_plus_delta = torch.einsum('nlds, nls -> nld', sampled_features_scaled, attn_score)
            # sum해주는 과정. LK가 사라진다. LK에 대해서 전부 더해줬기 때문.

            # B, M, H, W, C_v
            Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.view(BS, self.M, H, W, self.C_v)
            # B, H, W, M, C_v
            Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.permute(0, 2, 3, 1, 4).contiguous()
            # B, H, W, M * C_v
            Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.view(BS, H, W, self.C_v * self.M)

            final_features = self.W_m(Attention_W_prim_x_plus_delta)

        return final_features, output

    def compute_sampling(self, w_prim, phi_ref, offsets, layer, h, w):  # 결국 주어진 layer에 대해서 sampling. 이 함수가 핵심인듯 싶다.
        offseted_features = []
        for k in range(self.K): # offsets -> [BS*M, L, K, H, W, 2] 여러 layer와 key에 대해 봐야함.
            phi_ref_plus_delta = phi_ref + offsets[:, layer, k, :, :, :]  # 제안하는 ref
            vgrid_x = 2.0 * phi_ref_plus_delta[:, :, :, 0] / max(w - 1, 1) - 1.0  # [-1, 1] 값 갖도록
            vgrid_y = 2.0 * phi_ref_plus_delta[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3) # 더한 후 scale 된 형태로 [BS, H, W, 2]

            sampled = F.grid_sample(w_prim, vgrid_scaled, mode='bilinear', padding_mode='zeros')  # 주어진 좌표의 피쳐맵들 샘플링
            offseted_features.append(sampled)

        return torch.stack(offseted_features, dim=3)  # [BS*M, K, C_v,  H, W]