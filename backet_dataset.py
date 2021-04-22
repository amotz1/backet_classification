


class BacketDataset:
    def __init__(self, mode, num_classes=3, dataset_path='/data', dim=(224, 244)):
        self.root = '/content/' + str(dataset_path) + '/' + mode + '/'
        self.classes = num_classes
        self.dim = dim
        self.backetx_dataset_dict = {'circle': 0, 'square': 1, 'triangle': 2}

        valid_file = '/content/Bucketx/valid_images_labels.txt'
        train_file = '/content/Bucketx/train_images_labels.txt'

        if mode == 'train':
            self.paths, self.labels = read_file_paths(train_file)
            self.transform = train_transform

        elif mode == 'valid':
            self.paths, self.labels = read_file_paths(valid_file)
            self.transform = valid_transform

        else:
            assert False, 'unspecified mode'

        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_tensor = self.load_image(self.root + self.paths[index], self.dim, agumentation=self.mode)
        label_tensor = torch.tensor(self.backetx_dataset_dict[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='valid'):
        if not os.path.exist(img_path):
            print('image not found'.format(img_path))

        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        image_tensor = self.transform(image)

        return image_tensor









