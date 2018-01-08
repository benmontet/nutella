from pyke import KeplerTargetPixelFile, KeplerQualityFlags
import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from scipy.ndimage import zoom
from skimage import transform
from scipy.ndimage.measurements import center_of_mass as cm

from scipy.optimize import minimize
from tqdm import tqdm


__all__ = ['PSFModel']


class data(object):
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

    def __init__(self, tpf, tstart, tfin, subsampling):
        self.tpf = tpf
        #self.grid = grid
        self.tstart = tstart
        self.tfin = tfin
        self.ss = subsampling
        self.pad = 0.0
        #self._prepare_grid()
        
        self._prepare_data()
        self.star_template = self.make_template()
        self.det_template = np.ones(((len(self.flux[0][:,0])*self.ss,len(self.flux[0][0])*self.ss)))
    

    def _prepare_grid(self):
        y, x = self.grid
        self.super_x = np.linspace(x[0], x[-1], self.subsampling * len(x))
        self.super_y = np.linspace(y[0], y[-1], self.subsampling * len(y))
        
    def _prepare_data(self):
        self.time = self.tpf.time[np.nansum(self.tpf.flux, axis=(1,2)) != 0.0]
        self.ferr = self.tpf.flux_err[np.nansum(self.tpf.flux, axis=(1,2)) != 0.0]
        self.flux = self.tpf.flux[np.nansum(self.tpf.flux, axis=(1,2)) != 0.0]

        good = (self.time > self.tstart) & (self.time < self.tfin)
        self.time = self.time[good]
        self.flux = self.flux[good]
        self.ferr = self.ferr[good]
        
        self.lt = len(self.time)
        
        for i in range(self.lt):
            self.flux[i] -= np.nanmedian(self.flux[i])

    def estimate_centroids(self):
        x0 = np.zeros_like(self.time)
        y0 = np.zeros_like(self.time)
        temp_flux = self.flux + 0.0
        temp_flux[np.isnan(temp_flux)] = 0.0
        for i in range(self.lt):
            x0[i], y0[i] = cm(temp_flux[i])
            
        return x0, y0

    def make_template(self):
        """
        Makes a super resolution template.

        Returns
        -------
        super_tpf : ndarray
            Template
        """
        self.xc, self.yc = self.estimate_centroids()

        self.xd_ss = np.linspace(0, 6, 6*self.ss)
        self.yd_ss = np.linspace(0, 6, 6*self.ss)

        self.xp = np.arange(-3, 3, 1/self.ss)
        self.yp = np.arange(-3, 3, 1/self.ss)

        big_tpf = [transform.resize(self.flux[i], np.asarray(self.flux[i].shape)*self.ss, order=0) for i in range(self.lt)]
        super_tpf = np.zeros((self.lt, len(self.xp), len(self.yp)))
        used = np.zeros((self.lt, len(self.xp), len(self.yp)))
        for i in range(self.lt):
            super_tpf[i] = self.shift_model(big_tpf[i], -self.xc[i]+2.5, -self.yc[i]+2.5, 0.0)

        super_tpf[super_tpf == 0.0] = 'NaN'
        mean_super_tpf = np.nanmean(super_tpf, axis=0)
        mean_super_tpf[mean_super_tpf < 0] = 0.0
        mean_super_tpf[np.isnan(mean_super_tpf)] = 0.0
        norm_super_tpf = mean_super_tpf / np.sum(mean_super_tpf)
        
        return norm_super_tpf
    
    def shift_model(self, image, xshift, yshift, pad, full=True):
        if np.abs(xshift) < 1e-8:
            if np.abs(yshift) < 1e-8:
                return image
        pss = int(pad*self.ss)
        lx = len(self.xd_ss)
        ly = len(self.yd_ss)
        xshift_model = xshift*self.ss
        yshift_model = yshift*self.ss
        whole_x = int(np.floor(xshift_model))
        frac_x = np.mod(xshift_model, 1)
        whole_y = int(np.floor(yshift_model))
        frac_y = np.mod(yshift_model, 1)
        ur = frac_x*frac_y
        ul = (1-frac_x)*(frac_y)
        lr = frac_x*(1-frac_y)
        ll = (1-frac_x)*(1-frac_y)
        

        output = np.zeros((len(self.xd_ss), len(self.yd_ss)))
        if pss-whole_x <= 0.0:
            return output
        elif pss - whole_x + lx >= 74+lx:
            return output
        if pss-whole_y <= 0.0:
            return output
        elif pss - whole_y + ly >= 74+ly:
            return output

        output += ur*image[pss-whole_x-1:pss-whole_x-1 + lx, pss-whole_y-1:pss-whole_y + ly-1]
        output += ul*image[pss-whole_x:pss-whole_x + lx, pss-whole_y-1:pss-whole_y + ly-1]
        output += lr*image[pss-whole_x-1:pss-whole_x-1 + lx, pss-whole_y:pss-whole_y + ly]
        output += ll*image[pss-whole_x:pss-whole_x + lx, pss-whole_y:pss-whole_y + ly]


        return output
    
    def p_operator(self, image, xshift, yshift, pad):
        if np.abs(xshift) < 1e-8:
            if np.abs(yshift) < 1e-8:
                return image
        pss = int(pad*self.ss)
        lx, ly = image.shape[0], image.shape[1]
        xshift_model = xshift*self.ss
        yshift_model = yshift*self.ss
        whole_x = int(np.floor(xshift_model))
        frac_x = np.mod(xshift_model, 1)
        whole_y = int(np.floor(yshift_model))
        frac_y = np.mod(yshift_model, 1)


        ur = frac_x*frac_y
        ul = (1-frac_x)*(frac_y)
        lr = frac_x*(1-frac_y)
        ll = (1-frac_x)*(1-frac_y)




        output = np.zeros_like(self.det_template)

        if pss+whole_x < 0:
            return output
        if pss + whole_x + lx >= len(output[:,0]):
            return output
        if pss + whole_y < 0:
            return output
        if pss + whole_y + ly >= len(output[0]):
            return output

        output[pss + whole_x:pss + whole_x + lx, pss + whole_y:pss + whole_y + ly] += ll*image
        output[pss + whole_x+1:pss + whole_x+1 + lx, pss + whole_y:pss + whole_y + ly] += lr*image
        output[pss + whole_x:pss + whole_x + lx, pss + whole_y+1:pss + whole_y+1 + ly] += ul*image
        output[pss + whole_x+1:pss + whole_x+1 + lx, pss + whole_y+1:pss + whole_y+1 + ly] += ur*image

        return output
    
    def model_prime(self, f, dy, dx, template, extend=0.0):
        interp_again = self.p_operator(template, dx, dy, self.pad)
        return f * interp_again
    

    def model(self, f, dy, dx, template, small=True, extend=0.0):
        tmp = self.model_prime(f, dy, dx, template, extend)
        if small == True:
            tmp = tmp.reshape(len(self.flux[0][:,0]), self.ss, len(self.flux[0][0]), self.ss)
            return np.nansum(tmp, axis=(1, 3))
        else:
            return tmp*self.ss*self.ss
        

    def evaluate(self, p, data, uncert, st = None, dt = None):
        if st is None:
            st = self.star_template
        if dt is None:
            dt = self.det_template
        f, dy, dx = p
        r = np.multiply(self.model(f, dy, dx, st),self.model(1, 0, 0, dt)) - data
        #plt.imshow(r, interpolation='nearest', origin='lower')
        return np.nansum(r * r / uncert / uncert)
    
    def calc_results(self):
        self.results = []
        self.lnlike = []
        
        for i in tqdm(range(self.lt)):
            sol = minimize(self.evaluate, x0=(2500, self.yc[i]-2.5, self.xc[i]-2.5), args=(self.flux[i], self.ferr[i]), method='BFGS')
            #print(sol.fun)
            #print(sol.x)
            self.results.append(sol.x)
            self.lnlike.append(sol.fun)
            #print(lnlike)
        self.results = np.array(self.results)
    
    def plot_results(self):
        print(self.results)
        plt.plot(np.arange(self.lt), self.results[:,0]/np.median(self.results[:,0]))
        plt.show()
        plt.plot(np.arange(self.lt), self.results[:,1])
        plt.show()
        plt.plot(np.arange(self.lt), self.results[:,2])
        plt.show()
        plt.plot(self.results[:,2], self.results[:,1], '.')
        plt.show()

    def gradient_descent_dd(self):
        f_n = self.results[:,0]
        yc_n = self.results[:,1]
        xc_n = self.results[:,2]
        j = 0
        c = 2e-11/self.ss
        tmp_d = np.copy(self.det_template)

        
        models = [self.model(f_n[i], yc_n[i], xc_n[i], self.star_template)*self.model(1, 0, 0, tmp_d) for i in range(self.lt)]
        model_star = [self.model(f_n[i], yc_n[i], xc_n[i], self.star_template, small=False) for i in range(self.lt)]
        resids = (models - self.flux)/self.ferr/self.ferr
        interps = [transform.resize(resids[i], np.asarray(resids[i].shape)*self.ss, order=0) for i in range(self.lt)]

        dlnlike_dd = 2 * np.multiply(np.array(interps) , model_star)
        dlnlike_dd = np.nansum(dlnlike_dd, axis=0)
        # dlnlike_dd *= np.random.normal(1.0, 0.001, np.shape(dlnlike_dd))

        #plt.imshow(-dlnlike_dd, interpolation='nearest', origin='lower')
        #plt.colorbar()
        #plt.show()

        logL_before = np.nansum([self.evaluate((f_n[i], yc_n[i], xc_n[i]), self.flux[i], self.ferr[i], dt=tmp_d) for i in range(self.lt)])
        print(logL_before)

        while j < 400:
            tmp_o = np.copy(tmp_d)
            tmp_d = tmp_d - c * dlnlike_dd
            with np.errstate(invalid='ignore'):
                tmp_d[tmp_d < 0.0] = 0.0

            logL_after = np.nansum([self.evaluate((f_n[i], yc_n[i], xc_n[i]), self.flux[i], self.ferr[i], dt=tmp_d) for i in range(self.lt)])
            print(logL_after, c, j)
            if np.abs(c) < 1e-20:
                break
            if np.abs(logL_after - logL_before)/ logL_after < 1e-9:
                break
            if logL_after < logL_before:
                c *= 1.6
                models = [self.model(f_n[i], yc_n[i], xc_n[i], self.star_template)*self.model(1, 0, 0, tmp_d) for i in range(self.lt)]
                model_star = [self.model(f_n[i], yc_n[i], xc_n[i], self.star_template, small=False) for i in range(self.lt)]
                resids = (models - self.flux)/self.ferr/self.ferr
                interps = [transform.resize(resids[i], np.asarray(resids[i].shape)*self.ss, order=0) for i in range(self.lt)]

                dlnlike_dd = 2 * np.multiply(np.array(interps) , model_star)
                dlnlike_dd = np.nansum(dlnlike_dd, axis=0)
                #dlnlike_dd *= np.random.normal(1.0, 0.001, np.shape(dlnlike_dd))

                #plt.imshow(-dlnlike_dd, interpolation='nearest', origin='lower')
                #plt.colorbar()
                #plt.show()
                logL_before = logL_after
            else:
                tmp_d = np.copy(tmp_o)
                c *= -0.2


            j += 1
        self.det_template = tmp_d
        
    def gradient_descent_ds(self, plot=False):
        f_n = self.results[:,0]
        yc_n = self.results[:,1]
        xc_n = self.results[:,2]
        j = 0
        c = 1e-10/self.ss
        tmp_s = self.star_template
        i = 0
        models = [self.model(f_n[i], yc_n[i], xc_n[i], tmp_s)*self.model(1, 0, 0, self.det_template) for i in range(self.lt)]


        resids = (models - self.flux)/self.ferr/self.ferr
        big_resids = [transform.resize(resids[i], np.asarray(resids[i].shape)*self.ss, order=0) for i in range(self.lt)]
        interps = np.zeros((self.lt, len(self.xp), len(self.yp)))
        for i in range(self.lt):
            resids_multi = np.multiply(big_resids[i] , self.det_template)
            interps[i] = self.shift_model(resids_multi, -xc_n[i], -yc_n[i], self.pad)

            #plt.imshow(-interps[i], interpolation='nearest', origin='lower')
            #plt.show()


        dlnlike_ds = [2 * interps[i] * f_n[i] for i in range(self.lt)]
        dlnlike_ds = np.nanmean(dlnlike_ds, axis=0)
        if plot == True:
            plt.imshow(-dlnlike_ds, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.show()

        logL_before = np.nansum([self.evaluate((f_n[i], yc_n[i], xc_n[i]), self.flux[i], self.ferr[i], st=tmp_s) for i in range(self.lt)])

        print(logL_before)
        while j < 400:
            tmp_o = np.copy(tmp_s)
            tmp_s = tmp_s - c * dlnlike_ds
            tmp_s[tmp_s < 0] = 0.0
            logL_after = np.nansum([self.evaluate((f_n[i], yc_n[i], xc_n[i]), self.flux[i], self.ferr[i], st=tmp_s) for i in range(self.lt)])


            print(logL_after, c, j)
            if np.abs(c) < 1e-16:
                break
            if np.abs(logL_after - logL_before)/ logL_after < 1e-9:
                break
            if logL_after < logL_before:
                c *= 1.6
                models = [self.model(f_n[i], yc_n[i], xc_n[i], tmp_s)*self.model(1, 0, 0, self.det_template) for i in range(self.lt)]
                resids = (models - self.flux)/self.ferr/self.ferr
                big_resids = [transform.resize(resids[i], np.asarray(resids[i].shape)*self.ss, order=0) for i in range(self.lt)]
                interps = np.zeros((self.lt, len(self.xp), len(self.yp)))
                for i in range(self.lt):
                    resids_multi = np.multiply(big_resids[i] , self.det_template)
                    interps[i] = self.shift_model(resids_multi, -xc_n[i], -yc_n[i], self.pad)

                dlnlike_ds = 2 * interps
                dlnlike_ds = np.sum(dlnlike_ds, axis=0)
                logL_before = logL_after

            else:
                tmp_s = np.copy(tmp_o)
                c *= -0.2


            j += 1
        self.star_template = tmp_s



    
if __name__ == '__main__':
    tpf = KeplerTargetPixelFile('https://archive.stsci.edu/missions/k2/target_pixel_files/c1/201900000/12000/'
                             'ktwo201912552-c01_lpd-targ.fits.gz',
                            quality_bitmask=KeplerQualityFlags.CONSERVATIVE_BITMASK)

    
    a = data(tpf, tstart=2035, tfin=2035.5, subsampling=8)
    a.calc_results()
    for i in range(3):
        a.gradient_descent_ds()
        a.gradient_descent_dd()
        a.calc_results()
    a.plot_results()
    

