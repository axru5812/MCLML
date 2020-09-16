import sys

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

LYA = 1215.67
C =  299792  # km/s

class MclyaReader:
    def __init__(self):
        self.datapath = "/home/axel/PhD/MCLML/data/v21_grid/"
        # Parameters for the simulation at hand
        self.vexp = np.array(
            [
                0.0,
                20.0,
                50.0,
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700,
            ]
        )
        self.nhi = np.array(
            [
                16.0,
                18.0,
                18.5,
                19.0,
                19.3,
                19.6,
                19.9,
                20.2,
                20.5,
                20.8,
                21.1,
                21.4,
                21.7,
            ]
        )
        self.tau = np.array([0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        self.b = np.array([10.0, 20.0, 40.0, 80.0, 160.0])

    def read(self):
        pass

    def get_single_spectrum(self, ivexp, inh, ib, itau):
        print('V exp: ', self.vexp[ivexp])
        print('NHI: ', self.nhi[inh])
        print('b: ', self.b[ib])
        print('tau: ', self.tau[itau])
        Ncol = 7
        Nbyte_per_line = Ncol + 2
        filename = self._get_spec_filename(ivexp, inh, ib, itau)

        with open(filename, "rb") as fhin:
            tasdf = fhin.read(16)
            head = np.fromstring(tasdf, np.int32)
            Ntotesc, Ntot, NperFreq = head[1], head[2], head[3]
            Nbyte = Ntotesc * (4 * Ncol + 2 * 4)
            raw_dat = np.fromstring(fhin.read(Nbyte), np.float32)

        data = np.array(raw_dat.reshape((Ntotesc, Nbyte_per_line))[:, 2:], np.float64)[
            ::-1
        ]
        input_photons = data[:, 0]
        output_photons = data[:, 1]

        bins = np.unique(input_photons)

        intrinsic_spectrum = self.gen_intrinsic_spectrum(bins[:-1], EW=140, fwhm=400)

        plt.figure()
        plt.plot(bins[:-1], intrinsic_spectrum)
        plt.show()

        total_spectrum = np.zeros_like(bins[:-1])
        for i, input_bin in enumerate(bins[:-1]):
            # Calculate the escape fraction for the current bin
            fesc = len(input_photons[np.where(input_photons == input_bin)])
            fesc /= NperFreq
            sub_hist, _ = np.histogram(
                output_photons[np.where(input_photons == input_bin)], bins=bins
            )
            # Here we do weighting
            weighted_hist = sub_hist / fesc
            weighted_hist = weighted_hist * intrinsic_spectrum
            total_spectrum += weighted_hist
        plt.figure()
        plt.plot(bins[:-1], total_spectrum)
        plt.show()
        return bins[:-1], total_spectrum

    def gen_intrinsic_spectrum(self, binning, EW=100, fwhm=None):
        """
        """
        # from EW and width we can calculate the amplitude of the required
        # gaussian

        # Integral of a gaussian = amp * sigma * sqrt(2pi)
        # Equivalent width = Integral of a gaussian / continuum
        # A = EW * continuum / (sigma * sqrt(2pi)

        # Convert EW to km/s
        EW = EW / LYA * C
        print('EW [KMS] ', EW)

        continuum = np.ones_like(binning)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        A = EW / (sigma * np.sqrt(2 * np.pi))
        mean = 0

        line = self._gaussian(binning, A, mean, sigma)

        return line + continuum

    def _gaussian(self, x, amp, mean, sigma):
        return amp * np.exp(-1 * (x - mean)**2 / (2 * sigma**2))

    def _smooth(self, x, y, R=2000):
        """ Smooths an input spectra 
        """
        sigma = self._get_sigma(R)
        sampling = self._get_sampling(x)
        kern = sigma / sampling
        fl = gaussian_filter1d(y, kern)
        return x, fl

    def _get_sampling(self, x):
        diff = np.diff(x)
        return np.mean(diff)
    
    def _get_sigma(self, R):
        return C / R

    def _get_spec_filename(self, ivexp, inh, ib, itau):
        fn_base = "shell_v21_"
        fn_ext = ".ebin"
        filename = self.datapath + fn_base
        filename += self._int2char(ivexp)
        filename += self._int2char(inh)
        filename += self._int2char(ib)
        filename += self._int2char(itau)
        filename += fn_ext
        return filename

    def _int2char(self, x):
        """
        returns character based upon input integer. 
        0 = a
        1 = b
        2 = c
        ...
        """
        return chr(ord("a") + x)