import torch


class Validation_stacker(torch.utils.data.Dataset):
    def __init__(self,
                validation_path,
                dtype=torch.float32,
                device='cpu'):

        self.validation_photostack=torch.load(validation_path)[:100]
        self.sobel_x = 0.125 * torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=dtype, device=device).view((1, 1, 3, 3))
        self.sobel_y = 0.125 * torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=dtype, device=device).view((1, 1, 3, 3))
        
    def compute_gradients(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)
        G_x = torch.nn.functional.conv2d(tensor, self.sobel_x, padding='same')
        G_y = torch.nn.functional.conv2d(tensor, self.sobel_y, padding='same')
        gradient_images = torch.sqrt(G_x * G_x + G_y * G_y + 1e-8).squeeze()
        return gradient_images
    
    def __getitem__(self, idx):
        return self.validation_photostack[idx]

    def __len__(self):
        return len(self.validation_photostack)



