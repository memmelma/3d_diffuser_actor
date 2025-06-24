import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np

from diffuser_actor.utils.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from diffuser_actor.utils.encoder import Encoder
from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)

def small_random_rotation_matrix_batch(B, max_angle=3 * torch.pi / 180, device='cpu'):
    angles = torch.empty(B, 3, device=device).uniform_(-max_angle, max_angle)
    ax, ay, az = angles[:, 0], angles[:, 1], angles[:, 2]

    cx, cy, cz = torch.cos(angles.T)
    sx, sy, sz = torch.sin(angles.T)

    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)

    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cx, -sx], dim=1),
        torch.stack([zeros, sx, cx], dim=1)
    ], dim=1)

    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sy, zeros, cy], dim=1)
    ], dim=1)

    Rz = torch.stack([
        torch.stack([cz, -sz, zeros], dim=1),
        torch.stack([sz, cz, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)

    return torch.bmm(Rz, torch.bmm(Ry, Rx))

def augment_pointcloud_batch(pc, translation_range=0.03, rotation_range=3, noise_range=0.01):
    # pc: [B, T, 3, H, W]
    B, T, C, H, W = pc.shape
    assert C == 3, "Expected 3 channels for XYZ data"
    device = pc.device

    pc = pc.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, 3]
    pc_flat = pc.view(B, T, H * W, 3)  # [B, T, N, 3]

    # Translation: [B, 1, 1, 3] (same for all points in B)
    translation = torch.empty(B, 1, 1, 3, device=device).uniform_(-translation_range, translation_range)
    pc_flat = pc_flat + translation

    # Rotation: [B, 3, 3]
    R = small_random_rotation_matrix_batch(B, max_angle=rotation_range * torch.pi / 180, device=device)
    pc_flat = torch.matmul(pc_flat, R.transpose(1, 2).unsqueeze(1))  # [B, T, N, 3]

    # Gaussian noise: [B, T, N, 3] (per point)
    noise = torch.randn(B, T, H * W, 3, device=device) * noise_range
    pc_flat = pc_flat + noise

    pc = pc_flat.view(B, T, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
    return pc

def augment_rgb_sequence(imgs, brightness=0.2, contrast=0.2, color_jitter=0.1):
    # imgs: [B, T, C, H, W]
    device = imgs.device
    B, T, C, H, W = imgs.shape

    # Brightness: [B, 1, 1, 1]
    brightness_shift = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * brightness
    imgs = imgs + brightness_shift.unsqueeze(1)  # [B, T, C, H, W]

    # Contrast: [B, 1, 1, 1]
    contrast_scale = 1 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * contrast
    mean = imgs.mean(dim=(3, 4), keepdim=True)  # [B, T, C, 1, 1]
    imgs = (imgs - mean) * contrast_scale.unsqueeze(1) + mean

    # Color jitter: [B, C, 1, 1]
    jitter = 1 + (torch.rand(B, C, 1, 1, device=device) * 2 - 1) * color_jitter
    imgs = imgs * jitter.unsqueeze(1)  # [B, T, C, H, W]

    return imgs.clamp(0, 1)

class DiffuserJointer(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
                 joint_loc_bounds=None,
                 augment_pcd=False,
                 augment_rgb=False,
                 diffusion_timesteps=100,
                 nhist=3,
                 loss_weights=[30, 1],
                 unnormalize_loss=False,
                 relative=False,
                 traj_relative=False,
                 lang_enhanced=False,
                 num_attn_heads=6):
        super().__init__()
        self._relative = relative
        self._traj_relative = traj_relative
        self.use_instruction = use_instruction
        self.augment_pcd = augment_pcd
        self.augment_rgb = augment_rgb
        self.unnormalize_loss = unnormalize_loss
        self.encoder = Encoder(
            backbone=backbone,
            image_size=image_size,
            num_attn_heads=num_attn_heads,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            num_attn_heads=num_attn_heads,
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            nhist=nhist,
            lang_enhanced=lang_enhanced
        )
        self.trajectory_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        self.n_steps = diffusion_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds) if gripper_loc_bounds is not None else None
        self.joint_loc_bounds = torch.tensor(joint_loc_bounds) if joint_loc_bounds is not None else None
        self.loss_weights = loss_weights

    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper, augment_pcd=True, augment_rgb=True):
        
        augmented_pcd = visible_pcd.clone()
        if augment_pcd:
            augmented_pcd = augment_pointcloud_batch(augmented_pcd, translation_range=0.03, rotation_range=3, noise_range=0.0)

        augmented_rgb = visible_rgb.clone()
        if augment_rgb:
            augmented_rgb = augment_rgb_sequence(augmented_rgb, brightness=0.3, contrast=0.3, color_jitter=0.2)
        
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            augmented_rgb, augmented_pcd
        )
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]


        # from utils.meshcat import create_visualizer, visualize_pointcloud
        # vis = create_visualizer()
        
        # visualize_pointcloud(
        #     vis, 'visible_pcd',
        #     pc=visible_pcd[0,0].permute(1,2,0).reshape(-1, 3).cpu().numpy(),
        #     color=visible_rgb[0,0].permute(1,2,0).reshape(-1, 3).cpu().numpy() * 255,
        #     size=0.01
        # )
        # visualize_pointcloud(
        #     vis, 'augmented_pcd',
        #     pc=augmented_pcd[0,0].permute(1,2,0).reshape(-1, 3).cpu().numpy(),
        #     color=augmented_rgb[0,0].permute(1,2,0).reshape(-1, 3).cpu().numpy() * 255,
        #     size=0.01
        # )
        # points = context[0, :, :3].cpu().numpy()
        # visualize_pointcloud(
        #     vis, 'compressed_pcd',
        #     pc=points,
        #     color=np.array([255, 0, 0]),
        #     size=0.02
        # )
        # import IPython; IPython.embed()


        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats
            )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context)
        )
        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos  # sampled visual features
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.trajectory_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.trajectory_noise_scheduler.timesteps[0])
        noisy_condition_data = self.trajectory_noise_scheduler.add_noise(
            condition_data[..., :7], noise[..., :7], noise_t
        )
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        # Iterative denoising
        timesteps = self.trajectory_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            trajectory = self.trajectory_noise_scheduler.step(
                out[..., :7], t, trajectory[..., :7]
            ).prev_sample
            
        trajectory = torch.cat((trajectory, out[..., 7:]), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        if self.gripper_loc_bounds is not None:
            pcd_obs = torch.permute(self.normalize_pos(
                torch.permute(pcd_obs, [0, 1, 3, 4, 2])
            ), [0, 1, 4, 2, 3])
        
        curr_gripper = self.normalize_joint_pos(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper, augment_pcd=False, augment_rgb=False
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # unnormalize position
        trajectory[:, :, :7] = self.unnormalize_joint_pos(trajectory[:, :, :7])
        
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

    def normalize_joint_pos(self, pos):
        pos_min = self.joint_loc_bounds[0].float().to(pos.device)#[:pos.shape[-1]]
        pos_max = self.joint_loc_bounds[1].float().to(pos.device)#[:pos.shape[-1]]
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_joint_pos(self, pos):
        pos_min = self.joint_loc_bounds[0].float().to(pos.device)#[:pos.shape[-1]]
        pos_max = self.joint_loc_bounds[1].float().to(pos.device)#[:pos.shape[-1]]
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relaative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # Relative Trajectory as Action Representation: https://arxiv.org/pdf/2402.10329
        if self._traj_relative:
            anchor = curr_gripper
            gt_trajectory = gt_trajectory - anchor

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            traj = self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper
            )
            # Relative Trajectory as Action Representation: https://arxiv.org/pdf/2402.10329
            if self._traj_relative:
                traj = traj + anchor
            return traj
        
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        
        gt_trajectory[:, :, :7] = self.normalize_joint_pos(gt_trajectory[:, :, :7])
        if self.gripper_loc_bounds is not None:
            pcd_obs = torch.permute(self.normalize_pos(
                torch.permute(pcd_obs, [0, 1, 3, 4, 2])
            ), [0, 1, 4, 2, 3])
        curr_gripper = self.normalize_joint_pos(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper, augment_pcd=self.augment_pcd, augment_rgb=self.augment_rgb
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.trajectory_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        noisy_trajectory = self.trajectory_noise_scheduler.add_noise(
            gt_trajectory[..., :7], noise[..., :7],
            timesteps
        )
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            pos = layer_pred[..., :7]

            if self.unnormalize_loss:
                pos_loss = F.l1_loss(
                    self.unnormalize_joint_pos(pos),
                    self.unnormalize_joint_pos(noise[..., :7]),
                reduction='mean'
            )
            else:
                pos_loss = F.l1_loss(pos, noise[..., :7], reduction='mean')
            loss = pos_loss
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 7:]
                openess_loss = F.binary_cross_entropy_with_logits(openess, gt_openess) * self.loss_weights[1]
                loss += openess_loss
            total_loss = total_loss + loss
        return {
            "total_loss": total_loss,
            "pos_noise_l1": pos_loss,
            "openess_bce": openess_loss
        }


class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 nhist=3,
                 lang_enhanced=False):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        
        # Encoders
        self.traj_encoder = nn.Linear(7, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 7)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        pos_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        return [torch.cat((pos_pred, openess_pred), -1)]

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_rel_context_pos,
                        instr_feats):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return position, openess

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features
