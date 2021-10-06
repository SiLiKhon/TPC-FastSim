import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_images', type=str)
    parser.add_argument('--output_name', type=str, default='plots.png')

    args = parser.parse_args()

    variables = [
            'crossing_angle',
            'dip_angle',
            'drift_length',
            'pad_coord_fraction',
            'time_bin_fraction',
    ]

    stats = [
            'Mean0',
            'Mean1',
            'Sigma0^2',
            'Sigma1^2',
            'Cov01',
            'Sum',
    ]

    img_path = Path(args.path_to_images)
    images = [[Image.open(img_path / f'{s} vs {v}_amp_gt_1.png') for v in variables] for s in stats]

    width, height = images[0][0].size

    new_image = Image.new('RGB', (width * len(stats), height * len(variables)))

    x_offset = 0
    for img_line in images:
        y_offset = 0
        for img in img_line:
            new_image.paste(img, (x_offset, y_offset))
            y_offset += img.size[1]
        x_offset += img.size[0]

    new_image.save(img_path / args.output_name)


if __name__ == '__main__':
    main()
