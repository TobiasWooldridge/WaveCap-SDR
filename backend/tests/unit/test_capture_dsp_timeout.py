import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wavecapsdr.capture import Capture, CaptureConfig, Channel, ChannelConfig
from wavecapsdr.devices.fake import FakeDriver


def test_process_channels_parallel_timeout(monkeypatch):
    cfg = CaptureConfig(
        id="c1",
        device_id="fake0",
        center_hz=1_000_000.0,
        sample_rate=1_000_000,
    )
    capture = Capture(cfg=cfg, driver=FakeDriver())
    chan_cfg = ChannelConfig(id="ch1", capture_id="c1", mode="wbfm")
    channel = Channel(chan_cfg)
    channel.start()
    capture._channels[channel.cfg.id] = channel

    def slow_dsp(_samples, _sample_rate, _cfg):
        time.sleep(0.1)
        return None, {}

    import wavecapsdr.capture as capture_module

    monkeypatch.setattr(capture_module, "_process_channel_dsp_stateless", slow_dsp)

    samples = np.zeros(1000, dtype=np.complex64)
    with ThreadPoolExecutor(max_workers=1) as executor:
        start = time.perf_counter()
        results = capture._process_channels_parallel(samples, executor, timeout=0.01)
        elapsed = time.perf_counter() - start

    assert len(results) == 1
    result_channel, result_audio = results[0]
    assert result_channel is channel
    assert result_audio is None
    assert elapsed < 0.2
