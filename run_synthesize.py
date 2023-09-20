
# python3.7
"""Contains the code to synthesize images from a pre-trained models.
"""
import os
import argparse
from tqdm import tqdm

import torch
from models import build_model
from utils.image_utils import postprocess_image, save_image
from utils.visualizers import HtmlVisualizer


def run_mapping(G, z, context, eot_ind=None):
    """Run mapping network of the generator."""
    with torch.no_grad():
        global_text, local_text = G.text_head(context, eot_ind=eot_ind)
        mapping_results = G.mapping(z,
                                    label=None,
                                    context=global_text)
    return mapping_results['wp'], local_text


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


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', type=str, default='',
                        help='Path to the pre-trained models.')
    parser.add_argument('text_prompt', type=str, default='',
                        help='The text prompt, support reading from a file '
                             'or just given a text prompt.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--syn_num', type=int, default=100,
                        help='Number of synthesized images.')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution of the model output.')
    parser.add_argument('--results_dir', type=str, default='work_dirs/syn_res',
                        help='Results directory.')
    parser.add_argument('--seed', type=int, default=4,
                        help='Random seed.')
    parser.add_argument('--trunc_layers', type=int, default=None,
                        help='Number of layers to perform truncation.')
    parser.add_argument('--loop_mapping', type=int, default=16,
                        help='Loop number for getting average for wp.')
    parser.add_argument('--save_name', type=str, default='0',
                        help='Name to help save the file.')
    parser.add_argument('--num_z', type=int, default=3,
                        help='Number of z for each text prompt.')
    parser.add_argument('--save_png', action='store_true',
                        help='Whether or not to save the synthesized images.')
    parser.add_argument('--trunc_vals', type=str, default='0,0.05,0.1,0.15,0.2',
                        help='Default values for truncation.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    assert args.batch_size == 1, 'Current script only support bs equals to 1.'
    if os.path.exists(args.text_prompt):
        text_prompt = read_text(args.text_prompt)
    else:
        text_prompt = [args.text_prompt]
    syn_num = min(args.syn_num, len(text_prompt))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    clip_config = {'model_name':'ViT-L-14',
                   'pretrained':'openai',
                   'freeze_clip': True}
    clip = build_model('CLIPModel', **clip_config)

    g_config = {'resolution': 64,
                'image_channels':3,
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

    trunc_vals = parse_float(args.trunc_vals)
    if not trunc_vals:
        trunc_vals = [0, 0.05, 0.1, 0.15, 0.2]
    trunc_layers = args.trunc_layers
    if not trunc_layers:
        trunc_layers = G.num_layers

    visualizer_syn = HtmlVisualizer(image_size=args.resolution)
    visualizer_syn.reset(num_rows=syn_num * args.num_z,
                         num_cols=len(trunc_vals) + 1)
    head = ['Number Z']
    head += [f'trunc_val_{val_}' for val_ in trunc_vals]
    visualizer_syn.set_headers(head)
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_png:
        os.makedirs(f'{args.results_dir}/images', exist_ok=True)
    w_avg = G.w_avg.reshape(1, -1, G.w_dim)[:, :args.trunc_layers]
    for idx in tqdm(range(syn_num)):
        text = text_prompt[idx]
        _, enc_text, eot_ind = clip.encode_text(text=text, is_tokenize=True)
        if args.loop_mapping > 0:
            sum_wp = 0
            for _ in range(args.loop_mapping):
                z = torch.randn((args.batch_size, *G.latent_dim), device=device)
                tmp_res, _ = run_mapping(G, z, enc_text, eot_ind=eot_ind)
                sum_wp += tmp_res
            avg_wp = sum_wp / args.loop_mapping
            avg_wp = avg_wp[:, :args.trunc_layers]
        for z_i in range(args.num_z):
            z = torch.randn((args.batch_size, *G.latent_dim), device=device)
            row_ind = idx * args.num_z + z_i
            visualizer_syn.set_cell(row_ind, 0, text=f'z_{z_i}')
            for col_idx, trunc_psi in enumerate(trunc_vals):
                wp, local_text = run_mapping(G, z, enc_text, eot_ind=eot_ind)
                wp[:, :args.trunc_layers] = w_avg.lerp(wp[:, :args.trunc_layers], trunc_psi)
                if args.loop_mapping > 0:
                    wp[:, :args.trunc_layers] = avg_wp.lerp(wp[:, :args.trunc_layers], trunc_psi)
                fake_results = run_synthesize(G, wp, local_text)
                syn_imgs = fake_results['image'].detach().cpu().numpy()
                syn_imgs = postprocess_image(syn_imgs)
                visualizer_syn.set_cell(row_ind,
                                        col_idx + 1,
                                        text=text,
                                        image=syn_imgs[0])
                if args.save_png:
                    prefix = f'{args.results_dir}/images/text_{idx:04d}_z_'
                    save_path0 = f'{prefix}{z_i:02d}_psi_{trunc_psi:-2.1f}.png'
                    save_image(save_path0, syn_imgs[0])

    # Save result.
    save_name_syn = f'syn_{syn_num:04d}_seed_{args.seed}_{args.save_name}.html'
    save_path_syn = os.path.join(args.results_dir, save_name_syn)
    visualizer_syn.save(save_path_syn)


if __name__ == '__main__':
    main()
