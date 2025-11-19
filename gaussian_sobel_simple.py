import argparse
import numpy as np
import cv2
import os
import shutil

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

# INPUT_DIR = '../test/output'
# OUTPUT_DIR = '../test/output_enhanced'
ALPHA = 0.4

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing for Gaussian-Sobel Filtering.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
 
    parser.add_argument('--output_path', type=str,
                        help='output folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    
    # Optional filtering parameters
    parser.add_argument('--gauss_ksize', type=int, default=5,
                        help='Gaussian kernel size (must be odd).')
    parser.add_argument('--sobel_ksize', type=int, default=3,
                        help='Sobel kernel size (must be odd).')

    return parser.parse_args()

def enhance_depth_map(depth_map):
    depth_float = np.float64(depth_map)

    # Gaussian Filtering
    depth_smoothed = cv2.GaussianBlur(
        depth_float,
        (args.gauss_ksize, args.gauss_ksize),
        sigmaX=0
    )

    # Sobel Filtering
    grad_x = cv2.Sobel(depth_smoothed, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=args.sobel_ksize)
    grad_y = cv2.Sobel(depth_smoothed, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=args.sobel_ksize)

    # Gradient Magnitude
    # sqrt(gradx ^ 2 + grady ^ 2)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # normalize
    enhanced_depth_map = cv2.normalize(
        grad_magnitude,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_64F
    )

    return enhanced_depth_map

def process_map_simple(args):
    depth_file = args.image_path
    raw_depth_map = np.load(depth_file).squeeze()

    # H_in, W_in = raw_depth_map.shape
    # print(f"Input Shape (H x W): {H_in} x {W_in}")

    # Apply Gaussian-Sobel enhancement
    enhanced_depth_map = enhance_depth_map(raw_depth_map)
    # H_out, W_out = enhanced_depth_map.shape
    # print(f"Output Shape (H x W): {H_out} x {W_out}")

    # Try Fusing
    enhanced_depth_map = ALPHA * raw_depth_map + (1 - ALPHA) * enhanced_depth_map

    # Saving enhanced npy file
    # file_name = os.path.basename(depth_file)
    # output_name = file_name.replace('.npy', '_enhanced.npy')
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    output_name = os.path.splitext(os.path.basename(depth_file))[0]
    output_path = os.path.join(args.output_path, "{}_enhanced.npy".format(output_name))
    np.save(output_path, enhanced_depth_map)
    print(f"Enhanced npy file saved to: {output_path}")

    # Saving colormapped depth image
    enhanced_map_np = enhanced_depth_map.squeeze()
    vmax = np.percentile(enhanced_map_np, 95)
    normalizer = mpl.colors.Normalize(vmin=enhanced_map_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(enhanced_map_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    name_dest_im = os.path.join(args.output_path, "{}_enhanced.jpeg".format(output_name))
    im.save(name_dest_im)

    print(f"Enhanced jpeg saved to: {name_dest_im}")

if __name__ == '__main__':
    args = parse_args()
    process_map_simple(args)
