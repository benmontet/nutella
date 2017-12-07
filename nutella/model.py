import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RectBivariateSpline


__all__ = ['PSFModel']


class PSFModel(object):
    """
    Builds a PSF model given a pixel time series.

    Attributes
    ----------
    tpf : ndarray
        Pixel time series
    grid : (ndarray, ndarray)
        Tuple containing the vectors that define the grid for tpf
    subsampling : int
        Number of subdivisions inside one pixel
    """

    def __init__(self, tpf, grid, subsampling):
        self.tpf = tpf
        self.grid = grid
        self.subsampling = subsampling
        self._prepare_grid()

    def _prepare_grid(self):
        y, x = self.grid
        self.super_x = np.linspace(x[0], x[-1], self.subsampling * len(x))
        self.super_y = np.linspace(y[0], y[-1], self.subsampling * len(y))

    def estimate_centroids(self):
        yc, xc = [], []
        for i in range(len(self.tpf.shape[0])):
            _yc, _xc = center_of_mass(self.tpf[i])
            yc.append(_yc)
            xc.append(_xc)

        return yc, xc

    def make_template(self):
        """
        Makes a super resolution template.

        Returns
        -------
        super_tpf : ndarray
            Template
        """
        yc, xc = self.estimate_centroids()

        interp_objs = [RectBivariateSpline(self.super_x, self.super_y,
                                           tpf[i], kx=1, ky=1)
                       for i in range(tpf.shape[0])]

        recentered_tpf = []
        for i in range(len(tpf.shape[0])):
            interp_tpf = np.array([interp_objs[i](self.super_y + yc[i],
                                                  self.super_x + xc[i])])
            interp_tpf = interp_tpf / np.sum(interp_tpf)
            recentered_tpf.append(interp_tpf)
        recentered_tpf = np.array(recentered_tpf)

        super_tpf = np.mean(recentered_tpf, axis=0)
        super_tpf /= np.sum(super_tpf)

        return super_tpf

    def evaluate(self, f, dy, dx):
        """
        Evaluates the PSF value on the data domain.

        Parameters
        ----------
        f : float
            Total flux
        dy : float
            Center row
        dx : float
            Center column

        Returns
        -------
        psf : ndarray
            PSF evaluated on the data domain
        """
        template = self.make_template()
        interp_obj = RectBivariateSpline(self.super_x, self.super_y, template,
                                         kx=1, ky=1)
        return f * interp_obj(self.super_y - dy, self.super_x - dx)
