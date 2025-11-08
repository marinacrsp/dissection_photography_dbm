import glob
import os
import numpy as np
import torch
import nibabel as nib
from .photo_utils import myzoom_torch, gaussian_blur_3d, get_noninteger_coronal_slice_batch, get_noninteger_coronal_slice, fast_3D_interp_torch
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import gc


class GetSlice:
    def __init__(self, bsz):
        if bsz > 1:
            self.slice_func = get_noninteger_coronal_slice_batch
            
        else:
            self.slice_func = get_noninteger_coronal_slice
            print(f'Training with batchsize {bsz}')

    def __call__(self, volume, idx):
        return self.slice_func(volume, idx)


class Photosynth(torch.utils.data.Dataset):
    def __init__(self, datadir,
                spacing_limits=[3, 12], # sandwich size [2, 12]
                loss='l1',
                nvols=1,
                labels_to_kill=[3,4],
                siz=[160, 160],
                real_mix_prob=0.25,
                max_rotation=15,
                max_shear=0.2,
                max_scaling=0.2,
                nonlin_scale_min=0.03,
                nonlin_scale_max=0.06,
                nonlin_std_max=4,
                bf_scale_min=0.02,
                bf_scale_max=0.04,
                bf_std_min=0.1,
                bf_std_max=0.6,
                gamma_std=0.1,
                sigma_blur_min=0.1,
                sigma_blur_max=0.75,
                dtype=torch.float32,
                local_batch_size = 5,
                n_slices_per_sandwich = 1,
                provide_2d_gradients=True,
                device='cpu'):
        self.nvolumes = nvols
        self.datadir = datadir
        self.spacing_limits = spacing_limits
        self.loss = loss
        self.labels_to_kill = labels_to_kill
        self.siz = siz
        self.real_mix_prob = real_mix_prob
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scaling = max_scaling
        self.nonlin_scale_min = nonlin_scale_min
        self.nonlin_scale_max = nonlin_scale_max
        self.nonlin_std_max = nonlin_std_max
        self.bf_scale_min = bf_scale_min
        self.bf_scale_max = bf_scale_max
        self.bf_std_min = bf_std_min
        self.bf_std_max = bf_std_max
        self.gamma_std = gamma_std
        self.sigma_blur_min = sigma_blur_min
        self.sigma_blur_max = sigma_blur_max
        self.provide_2d_gradients = provide_2d_gradients
        self.n_slices_per_sandwich = n_slices_per_sandwich
        self.dtype = dtype
        self.device = device

        self.bsz = local_batch_size

        self.sobel_x = 0.125 * torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=dtype, device=device).view((1, 1, 3, 3))
        self.sobel_y = 0.125 * torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=dtype, device=device).view((1, 1, 3, 3))

        # Collect list of available images, per dataset
        datasets = []
        g = glob.glob(os.path.join(datadir, '*' + 'T1w.nii')) #Load the MRI images
        for i in range(len(g)):
            filename = os.path.basename(g[i])
            dataset = filename[:filename.find('.')]
            found = False
            for d in datasets:
                if dataset==d:
                    found = True
            if found is False:
                datasets.append(dataset)
        print('Found ' + str(len(datasets)) + ' datasets with ' + str(len(g)) + ' scans in total')
        names = []
        for i in range(len(datasets)):
            names.append(glob.glob(os.path.join(datadir, datasets[i] + '.*' + 'T1w.nii')))
        
        # flatten names
        self.data_list = [item for sublist in names for item in sublist]

        # Get resolution and maximum modeled distance for white / pial surfaces
        aux = nib.load(names[0][0])
        res_training_data = np.sum(aux.affine ** 2, axis=0)[:-1]

        # to cover the whole hemisphere
        # I guess this creates the dissection photos
        gensize = np.array([siz[0], 256, siz[1]]).astype(int)
        self.gensize = gensize

        self.get_slice = GetSlice(self.bsz)

        with torch.no_grad():
            # prepare grid
            print('Preparing grid...')
            xx, yy, zz = np.meshgrid(range(gensize[0]), range(gensize[1]), range(gensize[2]), sparse=False, indexing='ij')
            xx = torch.tensor(xx, dtype=self.dtype, device=self.device)
            yy = torch.tensor(yy, dtype=self.dtype, device=self.device)
            zz = torch.tensor(zz, dtype=self.dtype, device=self.device)
            c = torch.tensor((np.array(gensize) - 1) / 2, dtype=self.dtype, device=self.device)
            self.xc = xx - c[0]
            self.yc = yy - c[1]
            self.zc = zz - c[2]

            # Array to kill background labels in photo mode if needed
            lut_kill = torch.arange(0, 10000, dtype=torch.int32, device=self.device)
            for lln in labels_to_kill:
                lut_kill[lln] = 0
            self.lut_kill = lut_kill
            print('Generator is ready!')
    
    def compute_gradients(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)
        G_x = torch.nn.functional.conv2d(tensor, self.sobel_x, padding='same')
        G_y = torch.nn.functional.conv2d(tensor, self.sobel_y, padding='same')
        gradient_images = torch.sqrt(G_x * G_x + G_y * G_y + 1e-8).squeeze(1)
        return gradient_images

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        inputs, gradient_images, dists, outputs = [], [], [], []
        for _ in range(self.nvolumes):
            jidx = np.random.randint(len(self.data_list)) # this is to sample randomly volumes 
            t1 = self.data_list[jidx]
            t2 = self.data_list[jidx][:-7] + 'T2w.nii'
            flair = self.data_list[jidx][:-7] + 'FLAIR.nii'
            generation_labels = self.data_list[jidx][:-7] + 'generation_labels.nii'
            segmentation_labels = self.data_list[jidx][:-7] + 'brainseg.nii'
            spacing_simulation = self.spacing_limits[0] + np.random.rand() * (self.spacing_limits[1] - self.spacing_limits[0])
            t2 = t2 if os.path.isfile(t2) else None
            flair = flair if os.path.isfile(flair) else None

            # Load generation labels off the bat
            Gimg = nib.load(generation_labels)

            # sample affine deformation
            rotations = (2 * self.max_rotation * np.random.rand(3) - self.max_rotation) / 180.0 * np.pi
            shears = (2 * self.max_shear * np.random.rand(3) - self.max_shear)
            scalings = 1 + (2 * self.max_scaling * np.random.rand(3) - self.max_scaling)
            scaling_factor_distances = np.prod(scalings) ** .33333333333 # we divide distance maps by this, not perfect, but better than nothing
            A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=self.device)

            # sample center
            c2 = torch.tensor((np.array(Gimg.shape[0:3]) - 1)/2, dtype=self.dtype, device=self.device)

            # sample nonlinear deformation (photos a bit special)
            nonlin_scale = self.nonlin_scale_min + np.random.rand(1) * (self.nonlin_scale_max - self.nonlin_scale_min)
            siz_F_small = np.round(nonlin_scale * np.array(self.gensize)).astype(int).tolist()
            siz_F_small[1] = np.round(self.siz[1] / spacing_simulation).astype(int) # photos!
            nonlin_std = self.nonlin_std_max * np.random.rand()
            Fsmall = nonlin_std * torch.randn([*siz_F_small, 3], dtype=self.dtype, device=self.device)
            F = myzoom_torch(Fsmall, np.array(self.gensize) / siz_F_small, self.device)
            F[:, :, :, 1] = 0

            # deformed coordinates (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = self.xc + F[:, :, :, 0]
            yy1 = self.yc + F[:, :, :, 1]
            zz1 = self.zc + F[:, :, :, 2]
            xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
            yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
            zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]

            # Get the margins for reading images
            x1 = max(0, torch.floor(torch.min(xx2)).int().cpu().numpy())
            y1 = max(0, torch.floor(torch.min(yy2)).int().cpu().numpy())
            z1 = max(0, torch.floor(torch.min(zz2)).int().cpu().numpy())
            x2 = min(Gimg.shape[0], 1 + torch.ceil(torch.max(xx2)).int().cpu().numpy())
            y2 = min(Gimg.shape[1], 1 + torch.ceil(torch.max(yy2)).int().cpu().numpy())
            z2 = min(Gimg.shape[2], 1 + torch.ceil(torch.max(zz2)).int().cpu().numpy())
            xx2 -= int(x1)
            yy2 -= int(y1)
            zz2 -= int(z1)

            # Read in data
            G = torch.squeeze(torch.tensor(Gimg.get_fdata()[x1:x2, y1:y2, z1:z2], dtype=torch.int, device=self.device))
            S = torch.squeeze(torch.tensor(nib.load(segmentation_labels).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=torch.int, device=self.device))
            T1 = torch.squeeze(torch.tensor(nib.load(t1).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=self.dtype, device=self.device))
            T2 = None if t2 is None else torch.squeeze(torch.tensor(nib.load(t2).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=self.dtype, device=self.device))
            FLAIR = None if flair is None else torch.squeeze(torch.tensor(nib.load(flair).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=self.dtype, device=self.device))

            # Kill a bunch of labels
            M = self.lut_kill[S]>0
            S[~M] = 0
            G[~M] = 0
            T1[~M] = 0
            if T2 is not None:
                T2[~M] = 0
            if FLAIR is not None:
                FLAIR[~M] = 0

            # normalize images for later mixing
            T1 /= torch.median(T1[M])
            T2 = None if T2 is None else (T2/torch.median(T2[M]))
            FLAIR = None if FLAIR is None else (FLAIR / torch.median(FLAIR[M]))

            # Sample Gaussian image
            mus = 25 + 200 * torch.rand(256, dtype=self.dtype, device=self.device)
            sigmas = 5 + 20 * torch.rand(256, dtype=self.dtype, device=self.device)
            # set the background to zero
            mus[0] = 0
            sigmas[0] = 0

            #  Crucial bit: partial volume!
            # 1 = lesion, 2 = WM, 3 = GM, 4 = CSF
            v = 0.02 * torch.arange(50).to('cpu')
            mus[100:150] = mus[1] * (1 - v) + mus[2] * v
            mus[150:200] = mus[2] * (1 - v) + mus[3] * v
            mus[200:250] = mus[3] * (1 - v) + mus[4] * v
            mus[250] = mus[4]
            sigmas[100:150] = torch.sqrt(sigmas[1]**2 * (1 - v) + sigmas[2]**2 * v)
            sigmas[150:200] = torch.sqrt(sigmas[2]**2 * (1 - v) + sigmas[3]**2 * v)
            sigmas[200:250] = torch.sqrt(sigmas[3]**2 * (1 - v) + sigmas[4]**2 * v)
            sigmas[250] = sigmas[4]

            SYN = mus[G] + sigmas[G] * torch.randn(G.shape, dtype=self.dtype, device=self.device)
            SYN[SYN < 0] = 0
            SYN /= torch.median(SYN[M])

            # cosmetic blurring
            # note that we don't worry blurring foreground with black foreground because we do that at test time anyway
            sigma = self.sigma_blur_min + (self.sigma_blur_max - self.sigma_blur_min) * np.random.rand()
            SYNblur = gaussian_blur_3d(SYN, sigma * np.ones(3), self.device, dtype=self.dtype)

            # Make random linear combinations
            if np.random.rand() < self.real_mix_prob:
                v = torch.rand(4)
                v[2] = 0 if T2 is None else v[2]
                v[3] = 0 if FLAIR is None else v[3]
                v /= torch.sum(v)
                HR = v[0] * SYNblur + v[1] * T1
                if T2 is not None:
                    HR += v[2] * T2
                if FLAIR is not None:
                    HR += v[3] * FLAIR
            else:
                HR = SYNblur

            # deform everything at the same time!
            HRdef = fast_3D_interp_torch(HR, xx2, yy2, zz2, 'linear', self.device, dtype=self.dtype)
            
            # Gamma transform
            gamma = torch.tensor(np.exp(self.gamma_std * np.random.randn(1)[0]), dtype=float, device=self.device)
            HRgamma = 3.0 * (HRdef / 3.0) ** gamma

            # Bias field
            bf_scale = self.bf_scale_min + np.random.rand(1) * (self.bf_scale_max - self.bf_scale_min)
            siz_BF_small = np.round(bf_scale * np.array(self.gensize)).astype(int).tolist()
            siz_BF_small[1] = np.round(self.gensize[1]/spacing_simulation).astype(int)
            BFsmall = torch.tensor(self.bf_std_min + (self.bf_std_max - self.bf_std_min) * np.random.rand(1), 
                                dtype=self.dtype, device=self.device) * torch.randn(siz_BF_small, dtype=self.dtype, device=self.device)
            BFlog = myzoom_torch(BFsmall, np.array(self.gensize) / siz_BF_small, self.device)
            BF = torch.exp(BFlog)
            HRbf = HRgamma * BF

            # Generate random triplets!
            inputs_vol = torch.zeros([self.bsz, 2, self.siz[0], self.siz[1]], dtype=self.dtype, device=self.device) # xA, xB
            outputs_vol = torch.zeros([self.bsz, self.n_slices_per_sandwich, self.siz[0], self.siz[1]], dtype=self.dtype, device=self.device)
            dists_vol = torch.zeros([self.bsz, self.n_slices_per_sandwich, 2], dtype=self.dtype, device=self.device)

            spac = self.spacing_limits[0] + np.random.rand(self.bsz) * (self.spacing_limits[1] - self.spacing_limits[0])
            y_min = torch.tensor(0.5 * spac, dtype = self.dtype, device=self.device)
            y_max = torch.tensor(HRbf.shape[1] - 1.0 - 0.5 * spac, dtype = self.dtype, device=self.device)
            y = y_min + (y_max-y_min) * torch.rand(self.bsz, device=self.device)
            y1 = y - 0.5 * spac
            y2 = y + 0.5 * spac

            yi = y1 + (y2 - y1) * torch.rand(self.bsz, device=self.device)

            HRbf = self.normalize_11(HRbf)

            if self.bsz > 1:
                inputs_vol[:, 0] = self.get_slice(HRbf, y1).squeeze().permute([1,0,2])
                inputs_vol[:, 1] = self.get_slice(HRbf, y2).squeeze().permute([1,0,2])
                # outputs[:, 0] = get_noninteger_coronal_slice_batch(HRbf, yi).squeeze().permute([1,0,2])
                dists_vol = dists_vol.squeeze()
                outputs_vol[:, 0] = self.get_slice(HRbf, yi).squeeze().permute([1,0,2])
            else:
                inputs_vol[:, 0] = self.get_slice(HRbf, y1).permute([1,0,2])
                inputs_vol[:, 1] = self.get_slice(HRbf, y2).permute([1,0,2])
                outputs_vol[:, 0] = self.get_slice(HRbf, yi).permute([1,0,2])     

            dists_vol[..., 0] = yi - y1
            dists_vol[..., 1] = y2 - yi

            if self.provide_2d_gradients:
                gradient_images_vol = self.compute_gradients(outputs_vol)
                
            else:
                gradient_images_vol = None


            #convert nans to zeros
            if torch.isnan(inputs_vol).any():
                inputs_vol[torch.isnan(inputs_vol)] = 0
            if torch.isnan(outputs_vol).any():
                outputs_vol[torch.isnan(outputs_vol)] = 0
            if torch.isnan(dists_vol).any():
                dists_vol[torch.isnan(dists_vol)] = 0

            #check that outputs are not zeros
            inputs.append(inputs_vol)
            gradient_images.append(gradient_images_vol)
            dists.append(dists_vol)
            outputs.append(outputs_vol)


            # cleaning memory
            gc.collect()
        return [inputs, gradient_images, dists, outputs]

    
    def normalize_11(self, data): #Volume based
        data = 2*(data - data.min()) / (data.max() - data.min()).clip(1.e-10) - 1
        return data
    def normalize_01(self, data): #Volume based
        data = (data - data.min()) / (data.max() - data.min()).clip(1.e-10)
        return data

class PhotosynthModule(pl.LightningDataModule):
    def __init__(self, args, training_=True, device='cpu'):
        # Explicitly initialize LightningDataModule
        pl.LightningDataModule.__init__(self)
        self.args = args
        self.args.training_ = training_
        self.device = device
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = Photosynth(self.args.data_path, 
                                        device=self.device, 
                                        local_batch_size=self.args.local_batch_size # this is the number of slices that will get generated
                                        )
        return self.train_dataset

    def train_dataloader(self):
        # Generate 1 volume each time
        return DataLoader(self.train_dataset, 
        batch_size=1,#1, 
        shuffle=True, 
        num_workers=self.args.num_workers, 
        pin_memory=True, 
        drop_last=False,)

    def val_dataloader(self): ### TODO
        raise NotImplementedError

#######################
# Auxiliary functions #
#######################

def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A
