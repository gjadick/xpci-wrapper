#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:16:00 2024

@author: gjadick

Functions for simulating propagation-based x-ray phase-contrast imaging.

"""

from xscatter import get_dbm_mix
from phasetorch.mono.sim import simdata
from skimage.transform import downscale_local_mean
import numpy as np
from numpy.random import poisson
from scipy.signal import convolve2d


PROCESSES = 1
DEVICE = 'cpu'


def lorentzian2d(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    """
    gamma = fwhm/2
    X, Y = np.meshgrid(x, y)
    kernel = gamma / (2 * np.pi * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / np.sum(kernel)
    return kernel


class Material:
    def __init__(self, matcomp, density, name=''):
        self.matcomp = matcomp
        self.density = density
        self.name = name
        self.energy = None
        self.delta = None
        self.beta = None
        self.mu = None
        self.tmap = None

    def get_dbm(self, energy):   # delta, beta, mu 
        d, b, m = get_dbm_mix(self.matcomp, energy, self.density)
        self.delta = d
        self.beta = b
        self.mu = m
        self.energy = energy
        return d, b, m
    
    def assign_tmap(self, tmap):  # depends on chosen geometry
        self.tmap = tmap
        if self.energy is not None:
            self.get_db_proj()
        return None

    def get_db_proj(self):  # must have called get_dbm and assign_tmap already!
        assert self.delta is not None
        assert self.beta is not None
        assert self.tmap is not None
        dproj = []
        bproj = []
        for i in range(len(self.energy)):
            dproj.append(self.tmap * self.delta[i])
            bproj.append(self.tmap * self.beta[i])
        self.delta_proj = np.array(dproj)
        self.beta_proj = np.array(bproj)
        return self.delta_proj, self.beta_proj
    

class Geometry:
    def __init__(self, Nx, dx, upx):
        self.Nx = Nx
        self.dx = dx   # pixel size
        self.upx = upx
        self.FOVx = Nx * dx
        self.Nx_upx = Nx * upx
        self.dx_upx = dx / upx
        self.psf = np.array([[1.0]])
        self.noise_percent = 0.0

    def assign_lorentzian_psf(self, fwhm, N_fwhm=3, upsample=False):
        if upsample:
            dx = self.dx_upx
        else:
            dx = self.dx
        fov = N_fwhm * fwhm    # to reduce memory, truncate FOV
        xcoords = np.arange(-fov/2, fov/2, dx) + dx/2
        kernel = lorentzian2d(xcoords, xcoords, fwhm)
        self.psf = kernel
        return kernel
    
    def assign_noise(self, noise_perc):
        self.noise_percent = noise_perc
        return None



class Spectrum:
    def __init__(self, energies, counts):
        self.E = np.array(energies)
        self.I0 = np.array(counts)   # photons per detector pixel -- not scaled to pixel size!!
        assert len(self.E) == len(self.I0)
        self.size = self.E.size


def tmap_rect(geometry, L, xc, yc, rx, ry, upsampled=True):
    """
    Get projected thickness map through a rectangular prism.
    L -- length [distance]
    """
    if upsampled:
        Nx = geometry.Nx_upx
        dx = geometry.dx_upx
    else:
        Nx = geometry.Nx
        dx = geometry.dx

    # convert coordinates to be units of indices (i.e. num pixels)
    ic = xc / dx
    jc = yc / dx
    ri = rx / dx
    rj = ry / dx
    assert (ic >= 0) and (ic < Nx)   # check in bounds center
    assert (jc >= 0) and (jc < Nx)
    assert (ri >= 0)  # edges can go out-of-bounds
    assert (rj >= 0) 

    imap, jmap = np.meshgrid(np.arange(Nx), np.arange(Nx)) 
    tmap = np.zeros([Nx, Nx], dtype=np.float32)
    tmap[(imap >= np.floor(ic - ri)) & (imap < np.ceil(ic + ri)) & (jmap >= np.floor(jc - rj)) & (jmap < np.ceil(jc + rj))] = L

    return tmap


def tmap_cylinder(geometry, L, xc, yc, rx, ry, upsampled=True):
    """get projected thickness map through an ellipsoidal cylinder
    """
    if upsampled:
        Nx = geometry.Nx_upx
        dx = geometry.dx_upx
    else:
        Nx = geometry.Nx
        dx = geometry.dx

    # convert coordinates to be units of indices (i.e. num pixels)
    ic = xc / dx
    jc = yc / dx
    ri = rx / dx
    rj = ry / dx
    assert (ic >= 0) and (ic < Nx)   # check in bounds center
    assert (jc >= 0) and (jc < Nx)
    assert (ri >= 0)  # edges can go out-of-bounds
    assert (rj >= 0) 

    imap, jmap = np.meshgrid(np.arange(Nx), np.arange(Nx)) 
    tmap = np.zeros([Nx, Nx], dtype=np.float32)
    tmap[((imap - ic)/ri)**2 + ((jmap - jc)/rj)**2 <= 1] = L

    return tmap


def tmap_ellipsoid(geometry, L, xc, yc, rx, ry, upsampled=True):
    """
    get projected thickness map through an ellipsoid
    """
    if upsampled:
        Nx = geometry.Nx_upx
        dx = geometry.dx_upx
    else:
        Nx = geometry.Nx
        dx = geometry.dx

    # convert coordinates to be units of indices (i.e. num pixels)
    ic = xc / dx
    jc = yc / dx
    ri = rx / dx
    rj = ry / dx
    assert (ic >= 0) and (ic < Nx)   # check in bounds center
    assert (jc >= 0) and (jc < Nx)
    assert (ri >= 0)  # edges can go out-of-bounds
    assert (rj >= 0) 

    imap, jmap = np.meshgrid(np.arange(Nx), np.arange(Nx)) 
    tmap = 1 - ((imap - ic)/ri)**2 - ((jmap - jc)/rj)**2  # init, includes negatives!
    tmap[tmap < 0] = 0  # avoid sqrt of negative errors
    tmap = L * np.sqrt(tmap)

    return tmap


def simdata_polychrom(materials, spectrum, geo, propdists, normalize=True,
                      processes=PROCESSES, device=DEVICE):  
    try:
        len(materials)
    except:
        materials = [materials]  # check, only one material
    Nmats = len(materials) 

    try:
        len(propdists)
    except:
        propdists = [propdists]  # check, only one propagation distance
    Ndists = len(propdists)

    # Create the delta and beta line integral projections at all spectrum energies. 
    delta_proj_polychrom = np.zeros([spectrum.size, geo.Nx_upx, geo.Nx_upx], dtype=np.float32)
    beta_proj_polychrom = np.zeros([spectrum.size, geo.Nx_upx, geo.Nx_upx], dtype=np.float32)
    for mat in materials:
        assert geo.Nx_upx == mat.tmap.shape[0]
        assert geo.Nx_upx == mat.tmap.shape[1]
        delta_proj, beta_proj = mat.get_db_proj()
        delta_proj_polychrom += delta_proj
        beta_proj_polychrom = beta_proj

    # Simulate the images at all propagation distances.
    imgs_all = np.zeros([Ndists, geo.Nx, geo.Nx], dtype=np.float32)
    for i in range(spectrum.size):
        energy = spectrum.E[i]
        delta_proj = delta_proj_polychrom[i]
        beta_proj = beta_proj_polychrom[i]

        # initial monochromatic simulation (all propdists at once)
        rads = simdata(delta_proj, beta_proj, geo.dx, energy, propdists, processes=PROCESSES, device=DEVICE)
        rads = downscale_local_mean(rads, factors=(1, 1, geo.upx, geo.upx))

        # iterate over the distances for post-processing
        for j in range(Ndists):
            img_mono = spectrum.I0[i] * rads[0,j]
            if geo.noise_percent > 0:
                img_mono += 0.01 * geo.noise_percent * np.sqrt(img_mono) * poisson(size=img_mono.shape)
            imgs_all[j,:,:] += img_mono
    
    # Convolve with detector PSF after all photons at all energies have been counted.
    for i in range(len(propdists)):
        imgs_all[i] = convolve2d(imgs_all[i], geo.psf, mode='same', boundary='fill', fillvalue=np.sum(spec.I0))

    if normalize:
        imgs_all /= np.sum(spec.I0)

    return imgs_all




if __name__ == '__main__':   # testing

    test_materials = False
    test_tmaps = False
    test_sim = False

    # detector geometry
    Nx = 36
    px_sz = 5e-6
    upx = 8
    geo = Geometry(Nx, px_sz, upx)
    geo.assign_noise(noise_perc=0.4)
    geo.assign_lorentzian_psf(fwhm=px_sz)

    # polychromatic x-ray spectrum
    E = np.array([14, 16, 18, 20], dtype=np.float32)
    I0 = np.array([1, 1.2, 1.8, 1.1], dtype=np.float32)
    spec = Spectrum(E, I0)

    # two materials
    mat1 = Material('H(11.2)O(88.8)', 1.0, 'water')
    mat2 = Material('Al(100.0)', 2.699, 'Al')
    materials = [mat1, mat2]
    mat1.get_dbm(spec.E)
    mat2.get_dbm(spec.E)

    if test_materials:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=[9, 3], dpi=150)
        for mat in materials:
            ax[0].set_title('$\\delta(E)$')
            ax[0].plot(mat.energy, mat.delta, label=mat.name)
            ax[1].set_title('$\\beta(E)$')
            ax[1].plot(mat.energy, mat.beta)
            ax[2].set_title('$\\mu(E)$')
            ax[2].plot(mat.energy, mat.mu)
        for axi in ax:
            axi.set_xlabel('energy [keV]')
            axi.set_yscale('log')
        fig.legend(loc='center right')
        fig.tight_layout()
        plt.show()

    if test_tmaps:

        L = 8
        xc = geo.FOVx / 2
        yc = geo.FOVx / 2
        rx = px_sz * 8
        ry = px_sz * 14

        tmap1 =  tmap_rect(geo, L, xc, yc, rx, ry)
        tmap2 =  tmap_cylinder(geo, L, xc, yc, rx, ry)
        tmap3 =  tmap_ellipsoid(geo, L, xc, yc, rx, ry)
        tmaps = [tmap1, tmap2, tmap3]

        # plot check
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(tmaps), dpi=150, figsize=[3*len(tmaps)+1, 3])
        for i in range(len(ax)):
            m = ax[i].imshow(tmaps[i])
            fig.colorbar(m, ax=ax[i])
        fig.tight_layout()
        plt.show()

    if test_sim:
        propdists = [10e-3, 50e-3]

        # create some tmaps for the materials
        xc = geo.FOVx / 2
        yc = geo.FOVx / 2
        rx = px_sz * 6
        ry = px_sz * 13
        materials[0].assign_tmap(tmap_cylinder(geo, 1e-3, xc, yc, rx, ry))
        materials[1].assign_tmap(tmap_ellipsoid(geo, 2e-4, xc, yc, rx, ry).T)

        # calc the polychromatic projections
        delta_proj = np.zeros([spec.size, geo.Nx_upx, geo.Nx_upx], dtype=np.float32)
        beta_proj = np.zeros([spec.size, geo.Nx_upx, geo.Nx_upx], dtype=np.float32)
        for mat in materials:
            delta_proj += mat.delta_proj
            beta_proj += mat.beta_proj

        test_monosim = False
        if test_monosim:  # check monochromatic phasetorch sim
            rads = simdata(delta_proj, beta_proj, geo.dx, spec.E[0], propdists, processes=PROCESSES, device=DEVICE)
            rads = downscale_local_mean(rads, factors=(1, 1, geo.upx, geo.upx))

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, len(propdists))
            try:
                len(axs)
            except:
                axs = [axs]
            for i in range(len(axs)):
                im = axs[i].imshow(rads[0,i])
                axs[i].set_title("Distance {}".format(propdists[i]))
                fig.colorbar(im, ax=axs[i], orientation="horizontal")
            fig.suptitle('Monochromatic test', fontweight='bold')
            plt.tight_layout()
            plt.show()

        test_polysim = True
        if test_polysim:
            rads = simdata_polychrom(materials, spec, geo, propdists)

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, len(propdists)+1, figsize=[len(propdists)*2+3, 3])
            kw = {'cmap':'bwr', 'vmin':1-np.max(np.abs(1-rads)), 'vmax':1+np.max(np.abs(1-rads))}
            try:
                len(axs)
            except:
                axs = [axs]
            for i in range(len(axs)-1):
                im = axs[i].imshow(rads[i], **kw)
                axs[i].set_title("Distance {}".format(propdists[i]))
                fig.colorbar(im, ax=axs[i], orientation="horizontal")
            axs[-1].plot(spec.E, spec.I0)
            axs[-1].set_xlabel('[keV]')
            axs[-1].set_title('energy spectrum')
            fig.suptitle('Polychromatic test', fontweight='bold')
            plt.tight_layout()
            plt.show()


