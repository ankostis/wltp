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
from scipy.signal import butter, filtfilt

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
    return np.sqrt(x.dot(x)/x.size)

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

    def save_wave(self, signal, file_name):
        """Convert [-1, 1] floats --> bytes if (not already)."""
        if isinstance(signal, np.ndarray):
            dtype = f"<i{self._bits // 8}"
            signal = (signal * self._amplitude_scale).astype(dtype).tobytes()

        super().save_wave(signal, file_name)


@dataclass
class CycleAudio:
    #: (REQUIRED) The cycle's velocity trace.
    cycle: Collection[float] = field(repr=False)
    #: (REQUIRED) Desired duration of the "cycle clip".
    clip_duration_sec: int
    #: Derrived field.
    cycle_samples_count: int
    #: Derrived from given cycle, minus -1 (see :ref:`begin-annex`).
    cycle_duration_sec: int
    #: Derrived field.
    clip_samples_count: int
    #: Derrived field.
    cycle_sample_distance: int
    #: (optional) Derrived by :func:`running_mean_window_len` if not given.
    running_mean_window_len: int
    #: (Optional) Instance controlling audio sample-rate, etc, constructed
    #: with :attr:`amplitude` if not given.
    synth: Synthesizer = field(repr=False)
    #: A filter to eradicate "clicks" on the seams between the different cycle-samples.
    #: A Buteworth if not given.
    seam_filter: Synthesizer = field(repr=False)
    #: Pick samples from cycle every that often.
    #: If it is too small, sound-signal will be distorted ala "modem".
    min_audible_duration_sec: float = 0.02
    #: How loud?  Passed to :attr:`synth`, ignored if `synth` arg given.
    amplitude: float = 0.3
    #: Multiply :attr:`cycle` by that factor to produce the audible pitch.
    freq_shift: int = 10
    #: Butterworth cutoff frequency (against sampling rate, e.g. 3kHz for 44Khz sampling).
    cut_off_frequency: float = 1800
    #: (optional) if no :attr:`seam_filter` given, this is the Butterworth order
    #: applied with :func:`.filtfilt` (so this number is half of the effective order).
    seam_filter_order: int = 8

    def __init__(self, cycle, clip_duration_sec, **kw):
        vars(self).update(cycle=cycle, clip_duration_sec=clip_duration_sec, **kw)
        self.cycle_samples_count = len(self.cycle)
        self.cycle_duration_sec = self.cycle_samples_count - 1
        self.clip_samples_count = int(
            self.clip_duration_sec / self.min_audible_duration_sec
        )
        self.cycle_sample_distance = int(
            self.cycle_samples_count / self.clip_samples_count
        )
        if self.cycle_sample_distance < 1:
            self.cycle_sample_distance = 1

        if "synth" not in kw:
            self.synth: Synthesizer = Synthesizer(amplitude=self.amplitude)

        nyquist_freq = self.synth.framerate / 2
        # if "running_mean_window_len" not in kw:
        #     self.running_mean_window_len = int(
        #         calc_mean_window_len(self.synth.framerate, self.cut_off_frequency * nyquist_freq)
        #     )
        if "seam_filter" not in kw:
            self.seam_filter = butter(self.seam_filter_order, self.cut_off_frequency, fs=self.synth.framerate)

    def _filter_seam_clicks(self, signal: np.ndarray) -> np.ndarray:
        """Filter close to sampling frequency to remove "clicks" of signal seams. """
        # return running_mean(signal, self.running_mean_window_len)
        return filtfilt(*self.seam_filter, signal)
        # return signal

    def _limit_to_rms(self, signal: np.ndarray) -> np.ndarray:
        limit = rms(signal)
        signal[signal > limit] = limit
        signal[signal < -limit] = -limit

    def make_wav(self, fpath):
        # Arranged every :attr:`min_audible_duration_sec` timestep.
        clip_samples = self.cycle[:: self.cycle_sample_distance]

        clip = np.hstack(
            [
                self.synth.sine_pcm( # "sin" for V, "saw" for N
                    length=self.min_audible_duration_sec,
                    frequency=self.freq_shift * n + 1,
                )
                for n in clip_samples
            ]
        )
        self._limit_to_rms(clip)
        clip = self._filter_seam_clicks(clip)
        self.synth.save_wave(clip, fpath)


wltp_audio = CycleAudio(mdl.get_class_v_cycle("class3b"), 30)

## Set NEDC clip duration respectively to WLTP one.
#
nedc_cycle = nedc_cycle_data()["v_cycle"]
nedc_cycle_duration_sec = len(nedc_cycle) - 1
nedc_clip_duration_sec = (
    wltp_audio.clip_duration_sec
    * nedc_cycle_duration_sec
    / wltp_audio.cycle_duration_sec
)
nedc_audio = CycleAudio(nedc_cycle, nedc_clip_duration_sec)
