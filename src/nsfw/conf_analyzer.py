from torchvision import transforms as T


def create_transforms(transform_config):
    transform_list = []

    for transform_name, transform_params in transform_config.items():
        if transform_name == 'RandomResizedCrop':
            transform_list.append(T.RandomResizedCrop(tuple(transform_params['size'])))
        elif transform_name == 'RandomHorizontalFlip':
            transform_list.append(T.RandomHorizontalFlip(p=transform_params.get('p', 0.5)))
        elif transform_name == 'RandomRotation':
            transform_list.append(T.RandomRotation(degrees=transform_params['degrees']))
        elif transform_name == 'ColorJitter':
            transform_list.append(T.ColorJitter(
                brightness=transform_params.get('brightness', 0),
                contrast=transform_params.get('contrast', 0),
                saturation=transform_params.get('saturation', 0),
                hue=transform_params.get('hue', 0)
            ))
        elif transform_name == 'GaussianBlur':
            transform_list.append(T.GaussianBlur(
                kernel_size=transform_params['kernel_size']
            ))
        elif transform_name == 'Resize':
            transform_list.append(T.Resize(tuple(transform_params['size'])))
        elif transform_name == 'ToTensor':
            transform_list.append(T.ToTensor())
        elif transform_name == 'Normalize':
            transform_list.append(T.Normalize(
                mean=transform_params['mean'],
                std=transform_params['std']
            ))

    return T.Compose(transform_list)
