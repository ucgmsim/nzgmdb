"""
from scipy cookbook found at 
http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
and
http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
see also
http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

NOTE:
    butter with the sos option is implemented in scipy versions 16 and above. Currently
    the scipy version available on beatrice is 15. Use instead filter_design.py copied 
    from scipy_v17 source files found at
    
    https://github.com/scipy/scipy/blob/v0.17.0/scipy/signal/filter_design.py#L1824-L1895
"""
import filter_design as fd
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = fd.butter(order, [low, high], btype="band", analog=False, output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = fd.sosfilt(sos, data)
    return y


class ButterWorth:
    def __init__(self, data, fs=(1.0 / 0.005), ft=1.0, order=4):
        self.data = data
        self.fs = fs
        self.ft = ft
        self.order = order

    @staticmethod
    def filter(data, btype="lowpass", fs=(1.0 / 0.005), ft=1.0, order=4, output=None):
        """
        btype:
            'lowpass' or 'highpass'
        fs:
            sampling frequency
        ft:
            transition frequency
        Wn:
            For digital  filters Wn = ft/0.5*fs,
            where 0.5*fs is the Nyquist frequency
        """
        Nyquist_fq = 0.5 * fs
        Wn = ft / Nyquist_fq
        sos = fd.butter(order, Wn, btype=btype, analog=False, output="sos")

        y = fd.sosfilt(sos, data)
        if output is None:
            return y
        elif output == "sos":
            return y, sos

    def plot_freqResp(self, worN=512):
        """ """
        from scipy.signal import freqz
        from matplotlib import pylab as plt
        import numpy as np

        self.y_lf, self.sos_lf = self.filter(
            self.data, "lowpass", self.fs, self.ft, self.order, "sos"
        )
        self.b_lf, self.a_lf = fd.sos2tf(self.sos_lf)
        w_lf, h_lf = freqz(self.b_lf, self.a_lf, worN=worN)  # default worN=512)

        self.y_hf, self.sos_hf = self.filter(
            self.data, "highpass", self.fs, self.ft, self.order, "sos"
        )
        self.b_hf, self.a_hf = fd.sos2tf(self.sos_hf)
        w_hf, h_hf = freqz(self.b_hf, self.a_hf, worN=worN)  # default worN=512)

        w_lf, h_lf, w_hf, h_hf = (
            w_lf * 0.5 * self.fs * 0.5 / np.pi,
            np.abs(h_lf),
            w_hf * 0.5 * self.fs * 0.5 / np.pi,
            np.abs(h_hf),
        )
        plt.figure()
        plt.plot(w_lf, h_lf, label="low pass")
        plt.plot(w_hf, h_hf, label="high pass")
        plt.legend(loc="best")
        plt.xscale("log")

        fig = plt.figure()
        t = np.linspace(0.0, self.data.size * 0.005, self.data.size, endpoint=False)
        plt.plot(t, self.y_lf, label="low freq")
        plt.plot(t, self.y_hf, label="high freq")
        plt.legend(loc="best")
        #        plt.xscale('log')
        return


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = fd.butter(order, [low, high], btype="band", analog=False, output="ba")
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], "--", label="sqrt(0.5)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.grid(True)
    plt.legend(loc="best")

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + 0.11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label="Noisy signal")

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label="Filtered signal (%g Hz)" % f0)
    plt.xlabel("time (seconds)")
    plt.hlines([-a, a], 0, T, linestyles="--")
    plt.grid(True)
    plt.axis("tight")
    plt.legend(loc="upper left")

    from geoNet_file import GeoNet_File

    gf = GeoNet_File("/".join(["tests", "data", "20160214_001345_NBLC_20.V1A"]), vol=1)
    # from process import Process
    # gf_processed = Process(gf)
    # acc = gf_processed.comp_000
    acc = gf.comp_1st.acc
    filtered_acc = ButterWorth(acc)
    filtered_acc.plot_freqResp(worN=2000)
    plt.show()
