import torch
import numpy as np
import nibabel as nib
import os
from torch.nn.functional import conv3d

def myzoom_torch(X, factor, device, aff=None):

    if len(X.shape)==3:
        X = X[..., None]

    dtype = X.dtype

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], dtype=dtype, device=device)[:newsize[0]]
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], dtype=dtype, device=device)[:newsize[1]]
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], dtype=dtype, device=device)[:newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], dtype=dtype, device=device)

    for channel in range(X.shape[3]):
        Xc = X[:,:,:,channel]

        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], dtype=dtype, device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] +  wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], dtype=dtype, device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    if aff is not None:
        aff_new = aff.copy()
        for c in range(3):
            aff_new[:-1, c] = aff_new[:-1, c] / factor
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (0.5 - 0.5 / (factor * np.ones(3)))
        return Y, aff_new
    else:
        return Y

###############################

def viewVolume(x, aff=None):

    if aff is None:
        aff = np.eye(4)
    else:
        if type(aff) == torch.Tensor:
            aff = aff.cpu().detach().numpy()

    if type(x) is not list:
        x = [x]

    cmd = 'source /usr/local/freesurfer/nmr-dev-env-bash && freeview '

    c = 0
    for n in range(len(x)):
        if x[n] is not None:
            vol = x[n]
            if type(vol) == torch.Tensor:
                vol = vol.cpu().detach().numpy()
            vol = np.squeeze(np.array(vol))
            name = '/tmp/' + str(c) + '.nii.gz'
            c = c + 1
            MRIwrite(vol, aff, name)
            cmd = cmd + ' ' + name

    print(cmd + ' &')
    os.system(cmd + ' &')

###############################

def MRIwrite(volume, aff, filename, dtype=None):

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if aff is None:
        aff = np.eye(4)
    header = nib.Nifti1Header()
    nifty = nib.Nifti1Image(volume, aff, header)

    nib.save(nifty, filename)

###############################

def MRIread(filename, dtype=None, im_only=False, reorient=False):

    assert filename.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % filename

    x = nib.load(filename)
    if reorient:
        x = nib.as_closest_canonical(x)
    volume = x.get_fdata()
    aff = x.affine

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff


###############################

def make_gaussian_kernel(sigma, device, dtype):
    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2 * sl + 1, dtype=dtype, device=device)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()
    return kernel

###############################

def gaussian_blur_3d(input, stds, device, dtype=torch.float):

    blurred = input[None, None, :, :, :]
    if stds[0] > 0:
        kx = make_gaussian_kernel(stds[0], device=device, dtype=dtype)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1] > 0:
        ky = make_gaussian_kernel(stds[1], device=device, dtype=dtype)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2] > 0:
        kz = make_gaussian_kernel(stds[2], device=device, dtype=dtype)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return torch.squeeze(blurred)


def gaussian_blur_2d(input, stds, device, dtype=torch.float):

    blurred = input[None, None, :, :, :]
    if stds[0] > 0:
        kx = make_gaussian_kernel(stds[0], device=device, dtype=dtype)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1] > 0:
        ky = make_gaussian_kernel(stds[1], device=device, dtype=dtype)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2] > 0:
        kz = make_gaussian_kernel(stds[2], device=device, dtype=dtype)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return torch.squeeze(blurred)
##############################


def fast_3D_interp_torch(X, II, JJ, KK, mode, device, default_value=0.0, dtype=torch.float, return_linear_weights=False):
    if mode == 'nearest':
        ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & (II <= (X.shape[0] - 1)) & (JJ <= (X.shape[1] - 1)) & (KK <= (X.shape[2] - 1))
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape) == 3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], dtype=dtype, device=device)
        mask = (ok==False)
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[:, :, :, channel] = aux[IIr, JJr, KKr]
            Y[..., channel][mask] = default_value
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode == 'linear':
        ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & (II <= (X.shape[0] - 1)) & (JJ <= (X.shape[1] - 1)) & (KK <= (X.shape[2] - 1))
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape) == 3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], dtype=dtype, device=device)
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, dtype=dtype, device=device)
            Yc[ok] = c.type(dtype)
            Yc[~ok] = default_value
            Y[..., channel] = Yc

        if Y.shape[-1] == 1:
            Y = Y[..., 0]

    else:
        raise Exception('mode must be linear or nearest')

    if return_linear_weights:
        return Y, [ok, II.shape, fx, fy, fz, cx, cy, cz, wfx, wfy, wfz, wcx, wcy, wcz]
    else:
        return Y

##############
def get_noninteger_coronal_slice(X, y):
    yl = y.floor()
    yr = yl + 1
    wl = yr - y
    wr = 1 - wl
    I = wl * X[:, yl.long(), :] + wr * X[:, yr.long(), :]
    return I

def get_noninteger_coronal_slice_batch(X, y):
    yl = y.floor()
    yr = yl + 1
    wl = yr - y
    wr = 1 - wl
    wl_reshaped = wl.view(1, len(wl), 1)  # shape: [1, 16, 1]
    wr_reshaped = wr.view(1, len(wl), 1)
    I = wl_reshaped * X[:, yl.long(), :] + wr_reshaped * X[:, yr.long(), :]
    return I

###############################

def MRIwrite(volume, aff, filename, dtype=None):

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if aff is None:
        aff = np.eye(4)
    header = nib.Nifti1Header()
    nifty = nib.Nifti1Image(volume, aff, header)

    nib.save(nifty, filename)

###############################

def MRIread(filename, dtype=None, im_only=False, reorient=False):

    assert filename.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % filename

    x = nib.load(filename)
    volume = x.get_fdata()
    aff = x.affine

    if reorient: # brute force, who cares
        volume, aff = eugenios_closest_canonical(volume, aff)

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff


###############################
def eugenios_closest_canonical(volume, aff, return_ap_flip=False):
    aff_normalized = aff.copy()[:-1, :-1]
    for j in range(3):
        aff_normalized[:, j] = aff_normalized[:, j] / np.linalg.norm(aff_normalized[:, j])
    # Brute force, who cares, it's super fast
    permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    best_perm = None
    best_score = 0
    for p in range(len(permutations)):
        score = np.sum(np.abs(np.diag(aff_normalized[:, permutations[p]])))
        if score > best_score:
            best_score = score
            best_perm = permutations[p]
    aff2 = aff.copy()
    aff2[:-1, :-1] = aff[:-1, best_perm]
    if len(volume.shape) == 4:
        best_perm += [3]
    volume2 = volume.transpose(best_perm)
    ap_flip = (aff2[j, j] < 0)
    for j in range(3):
        if aff2[j, j] < 0:
            volume2 = np.flip(volume2, axis=j)
            aff2[:-1, -1] = aff2[:-1, -1] + aff2[:-1, j] * (volume2.shape[j] - 1.0)
            aff2[j, j] = -aff2[j, j]

    if return_ap_flip:
        return volume2, aff2, ap_flip
    else:
        return volume2, aff2