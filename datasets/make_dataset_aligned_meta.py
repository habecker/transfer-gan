import os

from PIL import Image
import cv2

def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.ppm') or filename.endswith('.pgm'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths

def open_image(image_path):
    if image_path[-3:] == 'ppm':
        return Image.fromarray(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))
    elif image_path[-3:] == 'pgm':
        return Image.fromarray(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    else:
        return Image.open(image_path)

def load_meta(dataset_folder):
    path = os.path.join(dataset_folder, 'meta.txt')
    meta = {}
    with open(path, 'r') as f:
        for line in f:
            dat = line.split(',')
            meta[dat[0]] = dat[1:]
    return meta

def align_images(meta, meta_target, a_file_paths, b_file_paths, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(meta_target, 'w') as f:
        for i in range(len(a_file_paths)):
            img_a = open_image(a_file_paths[i])
            img_b = open_image(b_file_paths[i])
            assert(img_a.size == img_b.size)

            identifier = os.path.basename(a_file_paths[i])[:-4]
            m = [str(i)] + [identifier] + meta[identifier]
            f.write(','.join(m))

            aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
            aligned_image.paste(img_a, (0, 0))
            aligned_image.paste(img_b, (img_a.size[0], 0))
            aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        help='Which folder to process (it should have subfolders testA, testB, trainA and trainB'
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_path
    print(dataset_folder)

    test_a_path = os.path.join(dataset_folder, 'testA')
    test_b_path = os.path.join(dataset_folder, 'testB')
    val_a_path = os.path.join(dataset_folder, 'valA')
    val_b_path = os.path.join(dataset_folder, 'valB')
    test_a_file_paths = get_file_paths(test_a_path)
    test_b_file_paths = get_file_paths(test_b_path)
    if os.path.exists(val_a_path) and os.path.exists(val_b_path):
        test_a_file_paths.extend(get_file_paths(val_a_path))
        test_b_file_paths.extend(get_file_paths(val_b_path))
    assert(len(test_a_file_paths) == len(test_b_file_paths))
    test_path = os.path.join(dataset_folder, 'test')

    train_a_path = os.path.join(dataset_folder, 'trainA')
    train_b_path = os.path.join(dataset_folder, 'trainB')
    train_a_file_paths = get_file_paths(train_a_path)
    train_b_file_paths = get_file_paths(train_b_path)
    assert(len(train_a_file_paths) == len(train_b_file_paths))
    train_path = os.path.join(dataset_folder, 'train')

    meta = load_meta(dataset_folder)

    align_images(meta, os.path.join(dataset_folder, 'meta_test.txt'), test_a_file_paths, test_b_file_paths, test_path)
    align_images(meta, os.path.join(dataset_folder, 'meta_train.txt'), train_a_file_paths, train_b_file_paths, train_path)
