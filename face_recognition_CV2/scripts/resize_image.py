import sys
from PIL import Image

def resize_image(in_path, out_path, max_width=800, max_height=800, quality=85):
    img = Image.open(in_path)
    img.thumbnail((int(max_width), int(max_height)), Image.LANCZOS)
    # Preserve format (JPEG quality param only applies to JPEG)
    save_kwargs = {}
    if img.format == 'JPEG':
        save_kwargs['quality'] = int(quality)
        save_kwargs['optimize'] = True
    img.save(out_path, **save_kwargs)

def main():
    if len(sys.argv) < 3:
        print('Usage: python resize_image.py <input> <output> [max_width] [max_height] [quality]')
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    max_w = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    max_h = int(sys.argv[4]) if len(sys.argv) > 4 else 800
    quality = int(sys.argv[5]) if len(sys.argv) > 5 else 85
    resize_image(in_path, out_path, max_w, max_h, quality)
    print(f'Resized {in_path} -> {out_path} (max {max_w}x{max_h}, q={quality})')

if __name__ == '__main__':
    main()
