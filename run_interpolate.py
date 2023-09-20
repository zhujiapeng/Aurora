
# python3.7
"""Contains the code to interpolate text embedding using a pre-trained models.
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch

from models import build_model
from utils.image_utils import postprocess_image
from utils.visualizers import HtmlVisualizer


def run_text_head(G, context, eot_ind=None):
    """Run the text head"""
    with torch.no_grad():
        global_text, local_text = G.text_head(context, eot_ind=eot_ind)
    return global_text, local_text


def run_mapping(G, z, global_text):
    """Run mapping network of the generator."""
    with torch.no_grad():
        mapping_results = G.mapping(z,
                                    label=None,
                                    context=global_text)
    return mapping_results['wp']


def run_synthesize(G, wp, local_text):
    """Run synthesis network of the generator."""
    with torch.no_grad():
        res = G.synthesis(wp, context=local_text)
    return res


def read_text(text_path):
    """Prepare snapshot text that will be used for evaluation."""
    print(f'Loading text from {text_path}')
    with open(text_path) as f:
        text = [line.strip() for line in f.readlines()]
    return text


def parse_float(arg):
    """Parse float number in string."""
    if not arg:
        return None
    arg = arg.split(',')
    arg = [float(i) for i in arg]
    return arg


def to_numpy(data):
    """Converts the input data to `numpy.ndarray`."""
    if isinstance(data, (int, float)):
        return np.array(data)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    raise TypeError(f'Not supported data type `{type(data)}` for '
                    f'converting to `numpy.ndarray`!')


def linear_interpolate(src_code,
                       dst_code,
                       layer_index=None,
                       steps=7):
    """Interpolate between the latent code and boundary."""
    assert (len(src_code.shape) == 3 and len(dst_code.shape) == 3 and
            src_code.shape[0] == 1 and dst_code.shape[0] == 1 and
            src_code.shape[1] == dst_code.shape[1])
    if not layer_index:
        layer_index = list(range(src_code.shape[1]))
    linspace = np.linspace(0.0, 1.0, steps)
    linspace = linspace.reshape([-1, 1, 1]).astype(np.float32)
    inter_code = src_code + linspace * (dst_code - src_code)
    is_inter = np.zeros(inter_code.shape, dtype=bool)
    is_inter[:, layer_index, :] = True
    inter_code = np.where(is_inter, inter_code, src_code)
    return inter_code


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', type=str, default='',
                        help='Path to the pre-trained models.')
    parser.add_argument('--src_prompt', type=str, default='',
                        help='The src text prompt, support reading from a file '
                             'or just given a text prompt.')
    parser.add_argument('--dst_prompt', type=str, default='',
                        help='The dst text prompt, support reading from a file '
                             'or just given a text prompt.')
    parser.add_argument('--num_src', type=int, default=10,
                        help='Number of src images.')
    parser.add_argument('--num_dst', type=int, default=10,
                        help='Number of dst images.')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution of the model output.')
    parser.add_argument('--results_dir', type=str, default='work_dirs/inter_res',
                        help='Results directory.')
    parser.add_argument('--seed', type=int, default=4,
                        help='Random seed.')
    parser.add_argument('--trunc_layers', type=int, default=None,
                        help='Number of layers to perform truncation.')
    parser.add_argument('--trunc_val', type=float, default=0.2,
                        help='Default value for truncation.')
    parser.add_argument('--loop_mapping', type=int, default=0,
                        help='Loop number for getting average for wp.')
    parser.add_argument('--save_name', type=str, default='0',
                        help='Name to help save the file.')
    parser.add_argument('--num_z', type=int, default=3,
                        help='Number of z for each text prompt.')
    parser.add_argument('--inter_layers', type=str, default=None,
                        help='The layers will be interpolated')
    parser.add_argument('--inter_step', type=int, default=9,
                        help='Number of interpolation steps. (default: 9)')
    parser.add_argument('--inter_g', action='store_true',
                        help='Whether to interp global code.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    assert args.src_prompt
    assert args.dst_prompt
    if os.path.exists(args.src_prompt):
        src_prompt = read_text(args.src_prompt)
    else:
        src_prompt = [args.src_prompt]
    if os.path.exists(args.dst_prompt):
        dst_prompt = read_text(args.dst_prompt)
    else:
        dst_prompt = [args.dst_prompt]
    num_src = min(args.num_src, len(src_prompt))
    num_dst = min(args.num_dst, len(dst_prompt))

    clip_config = {'model_name':'ViT-L-14',
                   'pretrained':'openai',
                   'freeze_clip': True}
    clip = build_model('CLIPModel', **clip_config)

    g_config = {'resolution': args.resolution,
                'image_channels': 3,
                'init_res': 4,
                'z_dim': 128,
                'w_dim': 1024,
                'mapping_fmaps': 1024,
                'label_dim': 0,
                'context_dim': 1024,
                'clip_out_dim': 768,
                'head_dim': 64,
                'embedding_dim': 1024,
                'use_text_cond': True,
                'num_layers_text_enc': 4,
                'use_w_cond': False,
                'use_class_label': False,
                'mapping_layers': 4,
                'fmaps_base': 16384,
                'fmaps_max': 1600,
                'num_adaptive_kernels': {"4":1,"8":1,"16":2,"32":4,"64":8},
                'num_block_per_res': {"4":3,"8":3,"16":3,"32":2,"64":2},
                'attn_resolutions': ['8', '16', '32', '64'],
                'attn_depth': {"8":2,"16":2,"32":2,"64":1},
                'attn_ch_factor': 1,
                'attn_gain': 0.3,
                'residual_gain': 0.4,
                'text_head_gain': 1.0,
                'zero_out': True,
                'fourier_feat': True,
                'l2_attention': True,
                'tie': False,
                'scale_in': False,
                'include_ff': True,
                'use_checkpoint': False,
                'checkpoint_res': ['8', '16', '32'],
                'mask_self': False,
                'conv_clamp': None,
                'mtm': True,
                'num_experts': {"8":4,"16":8,"32":16,"64":16},
                'ms_training_res': ['4','8','16','32','64'],
                'skip_connection': True}

    G = build_model('Text2ImageGenerator', **g_config)
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        print('Loading checkpoint from generator smooth!')
        G.load_state_dict(checkpoint['generator_smooth'])
    else:
        print('Loading checkpoint from generator!')
        G.load_state_dict(checkpoint['generator'])
    G = G.eval().to(device)

    trunc_layers = args.trunc_layers
    if not trunc_layers:
        trunc_layers = G.num_layers

    w_avg = G.w_avg.reshape(1, -1, G.w_dim)[:, :trunc_layers]
    num_rows = args.num_z * num_src * num_dst
    visualizer_syn = HtmlVisualizer(image_size=args.resolution)
    visualizer_syn.reset(num_rows=num_rows, num_cols=args.inter_step)
    head = [f'Step: {i:02d}' for i in range(1, args.inter_step + 1)]
    visualizer_syn.set_headers(head)
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    for z_idx in tqdm(range(args.num_z)):
        latent_z = torch.randn((1, *G.latent_dim), device=device)
        zs_avg = torch.randn((args.loop_mapping, *G.latent_dim), device=device)
        for src_idx in range(num_src):
            src_text = src_prompt[src_idx]
            _, src_enc_text, src_eot_ind = clip.encode_text(text=src_text,
                                                            is_tokenize=True)
            src_g_text, src_l_text = run_text_head(G,
                                                   src_enc_text,
                                                   eot_ind=src_eot_ind)
            for dst_idx in range(num_dst):
                dst_text = dst_prompt[dst_idx]
                _, dst_enc_text, dst_eot_ind = clip.encode_text(text=dst_text,
                                                                is_tokenize=True)
                dst_g_text, dst_l_text = run_text_head(G,
                                                       dst_enc_text,
                                                       eot_ind=dst_eot_ind)
                inter_g_codes = linear_interpolate(
                                   src_code=to_numpy(src_g_text)[:, np.newaxis],
                                   dst_code=to_numpy(dst_g_text)[:, np.newaxis],
                                   steps=args.inter_step)
                inter_l_codes = linear_interpolate(
                                    src_code=to_numpy(src_l_text),
                                    dst_code=to_numpy(dst_l_text),
                                    steps=args.inter_step)
                inter_g_codes = torch.from_numpy(inter_g_codes).to(device)
                inter_l_codes = torch.from_numpy(inter_l_codes).to(device)
                row_ind = (z_idx * num_src + src_idx) * num_dst + dst_idx
                for idx in range(inter_l_codes.shape[0]):
                    if args.loop_mapping > 0:
                        sum_wp = 0
                        for i in range(args.loop_mapping):
                            if args.inter_g:
                                tmp_res = run_mapping(G,
                                                      zs_avg[i:i+1],
                                                      inter_g_codes[idx:idx+1, 0])
                            else:
                                tmp_res = run_mapping(G,
                                                      zs_avg[i:i+1],
                                                      src_g_text)
                            sum_wp += tmp_res
                        avg_wp = sum_wp / args.loop_mapping
                        avg_wp = avg_wp[:, :trunc_layers]
                    if args.inter_g:
                        wp_i = run_mapping(G,
                                           latent_z,
                                           inter_g_codes[idx:idx+1, 0])
                    else:
                        wp_i = run_mapping(G, latent_z, src_g_text)
                    wp_i[:, :trunc_layers] = w_avg.lerp(wp_i[:, :trunc_layers], args.trunc_val)
                    if args.loop_mapping > 0:
                        wp_i[:, :trunc_layers] = avg_wp.lerp(wp_i[:, :trunc_layers], args.trunc_val)
                    img_inter = run_synthesize(G,
                                               wp_i,
                                               inter_l_codes[idx:idx+1])
                    img_inter = postprocess_image(to_numpy(img_inter['image']))
                    if idx == 0:
                        text = src_text
                    elif idx == args.inter_step - 1:
                        text = dst_text
                    else:
                        text = 'inter_res'
                    visualizer_syn.set_cell(row_ind,
                                            idx,
                                            text=text,
                                            image=img_inter[0])
    # Save result.
    save_name_syn = f'inter_seed_{args.seed}_{args.save_name}.html'
    save_path_syn = os.path.join(args.results_dir, save_name_syn)
    visualizer_syn.save(save_path_syn)


if __name__ == '__main__':
    main()
