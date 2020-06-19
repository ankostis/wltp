"""
Transform the velocity trace of a cycle into audible sound.

Example::

    wltp_audio.make_wav('/tmp/wltp.wav')
    nedc_audio.make_wav('/tmp/nedc.wav')
"""
import struct
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Collection, List

import numpy as np
import pandas as pd
from scipy import interpolate, signal

import chippy

from .. import datamodel as mdl
from .nedc import cycle_data as nedc_cycle_data


def running_mean(x, window_size):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def calc_mean_window_len(sample_rate, cut_off_frequency_hz) -> float:
    """
    Utility to design running-mean filter to eradicate "clicks".

    Example:

    >>> calc_mean_window_len(44100, 5000)
    4.032734391501626

    From https://dsp.stackexchange.com/a/14648/51023,
    as found in https://stackoverflow.com/a/35963967/548792
    """
    freqRatio = cut_off_frequency_hz / sample_rate
    return np.sqrt(0.196201 + freqRatio ** 2) / freqRatio


def rms(x):
    return np.sqrt(x.dot(x) / x.size)


class Synthesizer(chippy.Synthesizer):
    """
    Overridden to delay packing of float signal as bytes on saving

    and not on :meth:`pack_pcm_data()`.
    """

    def pack_pcm_data(self, wave_generator, length, envelope=None) -> np.ndarray:
        """OVERRIDDEN to return the enveloped ndarray[float32] (not bytes) of desired length."""
        num_bytes = int(self.framerate * length)
        wave = np.fromiter(wave_generator, dtype=np.float32, count=num_bytes)
        if not envelope:
            envelope = self.amplitude
        else:
            envelope = np.fromiter(envelope, dtype=np.float32, count=num_bytes)
        return wave * envelope

    def save_wave(self, sig, file_name):
        """Convert [-1, 1] floats --> bytes if (not already)."""
        if isinstance(sig, np.ndarray):
            dtype = f"<i{self._bits // 8}"
            sig = (sig * self._amplitude_scale).astype(dtype).tobytes()

        super().save_wave(sig, file_name)


@dataclass
class CycleAudio:
    #: (REQUIRED) The cycle's velocity trace.
    cycle: pd.Series = field(repr=False)
    #: (REQUIRED) Desired duration of the "cycle clip".
    clip_duration_sec: int
    #: Derrived field.
    cycle_samples_count: int
    #: Derrived from given cycle, minus -1 (see :ref:`begin-annex`).
    cycle_duration_sec: int
    #: (d)errived) How many V samples to pick from the cycle as audio pitches.
    #: Calculated based on :attr:`min_audible_duration_sec` and cycle-duration.
    v_samples_count: int
    #: Derrived field.
    cycle_sample_distance: int
    dtype: np.dtype
    bit_depth: int = 32
    sample_rate: int = 44100
    #: (optional) Derrived by :func:`running_mean_window_len` if not given.
    #: Pick samples from cycle every that often.
    #: If it is too small, sound-signal will be distorted ala "modem".
    #: If 0.0166665 for taking all samples of 1800sec cylce --> 30sec clip.
    min_audible_duration_sec: float = 0.1
    #: How loud?  Passed to :attr:`synth`, ignored if `synth` arg given.
    amplitude: float = 0.3
    #: FM carrier for the audible pitch.
    carrier_frequency: int = 300
    #: A Butterworth cutoff frequency (against sampling rate, e.g. 3kHz for 44Khz sampling)
    #: A filter to eradicate "notches" on the seams between the different cycle-samples.
    cut_off_frequency: float = 1800
    #: (optional) if no :attr:`seam_filter` given, this is the Butterworth order
    #: applied with :func:`.signal.filtfilt` (so this number is half of the effective order).
    lowband_filter_order: int = 8
    #: Limit signal to F x RMS(sig), to generate more frequencies
    #: (instead of triangular wave).
    rms_limiter_factor = 0.666

    def __init__(self, cycle, clip_duration_sec, **kw):
        vars(self).update(cycle=cycle, clip_duration_sec=clip_duration_sec, **kw)
        self.dtype = f"<i{self.bit_depth // 8}"
        self.cycle_samples_count = len(self.cycle)
        self.cycle_duration_sec = self.cycle_samples_count - 1
        self.v_samples_count = int(
            self.clip_duration_sec / self.min_audible_duration_sec
        )
        self.clip_samples_count = self.sample_rate * self.clip_duration_sec
        # TODO: DELETE; Rounding causes approximate clip-duration.
        self.cycle_sample_distance = min(
            1, int(self.cycle_samples_count / self.v_samples_count)
        )

        if "synth" not in kw:
            self.synth: Synthesizer = Synthesizer(amplitude=self.amplitude)

    def fm_wave(self, modulator):
        modulation = (
            2.0
            * np.pi
            * modulator.index
            * (self.carrier_frequency * (120 + 0.2 * modulator))
            / self.sample_rate
        )
        wave = self.amplitude * signal.square(modulation)
        wave[modulator == 0] = 0
        return wave

    def _filter_low_pass(self, sig: np.ndarray) -> np.ndarray:
        """Low-pass & nothc filter to remove "clicks" of signal seams. """
        # return running_mean(signal, self.running_mean_window_len)
        low_filter = signal.butter(
            self.lowband_filter_order,
            self.cut_off_frequency,
            fs=self.synth.framerate,
            output="sos",
        )
        return signal.sosfilt(low_filter, sig)

    def _limit_to_rms(self, sig: np.ndarray, f_rms) -> np.ndarray:
        limit = f_rms * rms(sig)
        sig[sig > limit] = limit
        sig[sig < -limit] = -limit

    def make_v_pitch_samples(self) -> pd.Series:
        cycle = self.cycle
        pchip = interpolate.PchipInterpolator(cycle.index, cycle.to_numpy())
        index = np.linspace(0, cycle.index[-1], self.v_samples_count)
        return pd.Series(pchip(index), index=index, name="v_pitch_samples")

    def make_clip_samples(self, v_samples) -> pd.Series:
        v_samples = v_samples.ewm(com=0.5).mean()
        pchip = interpolate.PchipInterpolator(v_samples.index, v_samples.to_numpy())
        index = np.linspace(0, self.cycle.index[-1], self.clip_samples_count)
        return pd.Series(pchip(index), index=index, name="clip_samples")

    def make_wav(self, fpath):
        ## Arranged every :attr:`min_audible_duration_sec` timestep.
        v_pitch_samples: pd.Series = self.make_v_pitch_samples()
        clip_samples: pd.Series = self.make_clip_samples(v_pitch_samples)

        clip = self.fm_wave(clip_samples)
        # self._limit_to_rms(clip, self.rms_limiter_factor)
        clip = self._filter_low_pass(clip)
        # clip = clip.to_numpy()
        self.synth.save_wave(clip, fpath)


wltp_audio = CycleAudio(mdl.get_class_v_cycle("class3b"), 30)

## Set NEDC clip duration respectively to WLTP one.
#
nedc_cycle = pd.Series(nedc_cycle_data()["v_cycle"])
nedc_cycle_duration_sec = len(nedc_cycle) - 1
nedc_clip_duration_sec = int(
    wltp_audio.clip_duration_sec
    * nedc_cycle_duration_sec
    / wltp_audio.cycle_duration_sec
)
nedc_audio = CycleAudio(nedc_cycle, nedc_clip_duration_sec)
