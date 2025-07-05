import cv2
import os
import argparse
import sys
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import crop_faces

def video_preprocess(dataset = "", root_dir = ""):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = os.listdir(dir)
        for fname in fnames:
            if fname.split('.')[0][-1] == ')':
                os.remove(os.path.join(dir, fname))
                print(f'---------  remove {os.path.join(dir, fname)} for endwith ([0-9])  ---------')
        for fname in sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0])):
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
        return images
    
    def make_dataset_from_video(video):
        images = []
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fname = f'{i:05d}'
            if ret:
                images.append((fname, frame))
        cap.release()
        return images
  
    if dataset == "VIPL":
        for p in sorted(os.listdir(root_dir)):
            if not p.startswith('p'):
                continue
            p_path = os.path.join(root_dir, p)
            for v in sorted(os.listdir(p_path)):
                v_path = os.path.join(p_path, v)
                for source in sorted(os.listdir(v_path)):
                    if source == 'source4':
                        print(f'jump over {v_path}/source4 -----------------')
                        continue
                    save_path = os.path.join(v_path, source, 'align_crop_pic')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    elif os.path.exists(os.path.join(v_path, source, 'pic')):
                        if len(os.listdir(save_path)) == len(os.listdir(os.path.join(v_path, source, 'pic'))):
                            print(f'already processed {save_path} -----------------')
                            continue

                    source_path = os.path.join(v_path, source)
                    video_path = os.path.join(source_path, 'video.avi')
                    print(f'processing {video_path}  ...')
                    
                    files = make_dataset_from_video(video_path)

                    image_size = 128
                    # scale = 1.0  # align only
                    scale = 0.8  # align + crop
                    center_sigma = 1.0
                    xy_sigma = 3.0
                    use_fa = False

                    crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

                    if crops is None:
                        print(f'too less face detected in video {video_path} -----------------')
                        continue

                    for i in range(len(crops)):
                        img = crops[i]
                        img.save(os.path.join(save_path, files[i][0] + '.png'))
                    print(f'generated {len(crops)} png images in file {save_path}')

    elif dataset == "UBFC":
        for idx, subject in enumerate(sorted(os.listdir(root_dir))):
            print(f'processing {subject} [{idx} / {len(os.listdir(root_dir))}] ...')
            subject_path = os.path.join(root_dir, subject)

            video_path = os.path.join(subject_path, '001vid.avi')
            save_path = os.path.join(subject_path, 'align_crop_pic')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            files = make_dataset_from_video(video_path)

            image_size = 128
            scale = 0.8
            center_sigma = 1.0
            xy_sigma = 3.0
            use_fa = False

            crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

            if crops is None:
                print(f'too less face detected in file {video_path} -----------------')
                continue

            for i in range(len(crops)):
                img = crops[i]
                img.save(os.path.join(save_path, files[i][0] + '.png'))
            print(f'generated {len(crops)} png images in file {save_path}')

    elif dataset == "PURE":
        date_list = sorted(os.listdir(root_dir))
        for date in date_list:
            # read video
            video_dir = os.path.join(root_dir, date, date)
            video_save_dir = os.path.join(root_dir, date, 'align_crop_pic')
            if not os.path.exists(video_save_dir):
                os.makedirs(video_save_dir)
            else:
                if len(os.listdir(video_save_dir)) == len(os.listdir(video_dir)):
                    print(f'already processed {video_save_dir} -----------------')
                    continue
            files = []
            for fname in sorted(os.listdir(video_dir)):
                if is_image_file(fname):
                    path = os.path.join(video_dir, fname)
                    fname = fname.split('.')[0]
                    files.append((fname, path))
                else:
                    raise ValueError(f'frame {fname} is not png')
            
            image_size = 128
            scale = 0.8
            center_sigma = 1.0
            xy_sigma = 3.0
            use_fa = False

            crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

            if crops is None:
                print(f'too less face detected in file {video_dir} -----------------')
                continue

            for i in range(len(crops)):
                img = crops[i]
                img.save(os.path.join(video_save_dir, files[i][0] + '.png'))
            print(f'generated {len(crops)} png images in file {video_save_dir}')

    elif dataset == "COHFACE":
        for px in sorted(os.listdir(root_dir)):
            px_path = os.path.join(root_dir, px)
            if not os.path.isdir(px_path) or px == 'protocols':
                continue
            for v_src in sorted(os.listdir(px_path)):
                v_path = os.path.join(px_path, v_src, 'data.avi')
                save_path = os.path.join(px_path, v_src, 'align_crop_pic')
                print(f'processing {v_path}  ...')
                cap = cv2.VideoCapture(v_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    if len(os.listdir(save_path)) == frame_count:
                        print(f'already processed {save_path} -----------------')
                        continue

                files = make_dataset_from_video(v_path)

                image_size = 128
                scale = 0.8
                center_sigma = 1.0
                xy_sigma = 3.0
                use_fa = False

                crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

                if crops is None:
                    print(f'too less face detected in video {v_path} -----------------')
                    continue

                for i in range(len(crops)):
                    img = crops[i]
                    img.save(os.path.join(save_path, files[i][0] + '.png'))
                print(f'generated {len(crops)} png images in file {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COHFACE', help='VIPL or UBFC or PURE or COHFACE')
    parser.add_argument('--dataset_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()
    video_preprocess(args.dataset, args.dataset_dir)
