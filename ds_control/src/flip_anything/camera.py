import time
from pathlib import Path


def _configure_sdk_logging(log_dir=None):
    from pyorbbecsdk import Context, OBLogLevel
    directory = Path(log_dir) if log_dir else Path(__file__).resolve().parents[3] / 'logs/sdk/ds_control'
    directory.mkdir(parents=True, exist_ok=True)
    Context.set_logger_to_file(OBLogLevel.WARNING, str(directory.resolve()))


def _intrinsic(value):
    return {k: getattr(value, k) for k in ("width", "height", "fx", "fy", "cx", "cy")}


def _distortion(value):
    return {k: getattr(value, k) for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")}


class OrbbecCamera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None
        self.started = False

    @staticmethod
    def enumerate(log_dir=None):
        from pyorbbecsdk import Context
        _configure_sdk_logging(log_dir)
        context = Context()
        devices = context.query_devices()
        result = []
        for i in range(devices.get_count()):
            info = devices.get_device_by_index(i).get_device_info()
            result.append({"name": info.get_name(), "serial": info.get_serial_number(),
                           "firmware": info.get_firmware_version()})
        return result

    def open(self):
        import numpy as np
        from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, AlignFilter, OBStreamType
        _configure_sdk_logging(self.cfg.get("log_dir"))
        self.context = Context()
        devices = self.context.query_devices()
        count = devices.get_count()
        if not count:
            raise RuntimeError("No Orbbec camera found")
        serial = self.cfg["serial"]
        if serial:
            device = devices.get_device_by_serial_number(serial)
        elif count == 1:
            device = devices.get_device_by_index(0)
        else:
            raise RuntimeError("Multiple cameras found; configure camera.serial explicitly")
        info = device.get_device_info()
        if "Gemini 336" not in info.get_name():
            raise RuntimeError(f"Expected Gemini 336, found {info.get_name()}")
        self.pipeline = Pipeline(device)
        config = Config()
        for sensor in (OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR):
            profile = self.pipeline.get_stream_profile_list(sensor).get_default_video_stream_profile()
            config.enable_stream(profile)
        self.aligner = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) if self.cfg["align_depth_to_color"] else None
        try:
            self.pipeline.enable_frame_sync()
            self.pipeline.start(config)
            self.started = True
            params = self.pipeline.get_camera_param()
            return {"name": info.get_name(), "serial": info.get_serial_number(),
                    "firmware": info.get_firmware_version(), "depth_aligned_to_color": self.aligner is not None,
                    "factory_camera_param": {"color_intrinsic": _intrinsic(params.rgb_intrinsic),
                        "depth_intrinsic": _intrinsic(params.depth_intrinsic),
                        "color_distortion": _distortion(params.rgb_distortion),
                        "depth_distortion": _distortion(params.depth_distortion),
                        "depth_to_color_rotation": np.asarray(params.transform.rot).reshape(3, 3).tolist(),
                        "depth_to_color_translation_mm": np.asarray(params.transform.transform).reshape(3).tolist()}}
        except BaseException:
            self.close()
            raise

    def capture(self, directory, index):
        # The Gemini 336 also intermittently delivers a corrupt MJPG payload on the first
        # frameset after start, which used to abort the whole session through the fatal
        # error path. A decode failure is transient, so retry the entire grab (files are
        # only written after a successful decode). Same class of defect as the incomplete
        # frameset retry below.
        last = None
        for _ in range(3):
            try:
                return self._capture_once(directory, index)
            except (RuntimeError, IOError, TimeoutError) as exc:
                last = exc
                time.sleep(0.25)
        raise last

    def _capture_once(self, directory, index):
        import cv2
        import numpy as np
        from pyorbbecsdk import OBFormat
        start = time.monotonic_ns()
        frames = None
        original_color = original_depth = None
        # The Gemini 336 intermittently delivers a frameset without one of the two
        # streams; retry instead of failing the whole capture (this used to abort the
        # session through the executor's fatal-error path).
        for _attempt in range(6):
            frames = self.pipeline.wait_for_frames(self.cfg["timeout_ms"])
            if frames is None:
                continue
            original_color, original_depth = frames.get_color_frame(), frames.get_depth_frame()
            if original_color is not None and original_depth is not None:
                break
            time.sleep(0.05)
        received = time.monotonic_ns()
        if frames is None:
            raise TimeoutError("Timed out waiting for RGB-D frames")
        if original_color is None or original_depth is None:
            raise RuntimeError("Incomplete source RGB-D frameset after retries")
        color_us, depth_us = original_color.get_timestamp_us(), original_depth.get_timestamp_us()
        color_system_us = original_color.get_system_timestamp_us()
        depth_system_us = original_depth.get_system_timestamp_us()
        if self.aligner is not None:
            aligned = self.aligner.process(frames)
            if aligned is None:
                raise RuntimeError("RGB-D alignment returned no frame")
            frames = aligned.as_frame_set()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is None or depth is None:
            raise RuntimeError("Incomplete RGB-D frameset")
        width, height = color.get_width(), color.get_height()
        data = np.asarray(color.get_data(), dtype=np.uint8)
        fmt = color.get_format()
        if fmt == OBFormat.MJPG:
            bgr = cv2.imdecode(data.reshape(-1), cv2.IMREAD_COLOR)
        elif fmt == OBFormat.RGB:
            bgr = cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.BGR:
            bgr = data.reshape(height, width, 3)
        elif fmt in (OBFormat.YUYV, OBFormat.UYVY):
            conversion = cv2.COLOR_YUV2BGR_YUY2 if fmt == OBFormat.YUYV else cv2.COLOR_YUV2BGR_UYVY
            bgr = cv2.cvtColor(data.reshape(height, width, 2), conversion)
        else:
            raise RuntimeError(f"Unsupported color format: {fmt}; select RGB/BGR/MJPG/YUYV/UYVY profile")
        if bgr is None:
            raise RuntimeError("Color image decode failed")
        if depth.get_format() != OBFormat.Y16:
            raise RuntimeError(f"Expected Y16 depth, got {depth.get_format()}")
        raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width()).copy()
        base = Path(directory) / f"frame_{index:06d}"
        if not cv2.imwrite(str(base) + "_color.png", bgr):
            raise IOError("Failed to save color image")
        np.save(str(base) + "_depth_raw.npy", raw, allow_pickle=False)
        return {"color_file": base.name + "_color.png", "depth_file": base.name + "_depth_raw.npy",
                "depth_scale_mm": depth.get_depth_scale(), "invalid_depth_raw": 0,
                "color_shape_hw": [height, width], "depth_shape_hw": list(raw.shape),
                "color_format": str(fmt), "aligned_to_color": self.aligner is not None,
                "color_device_us": color_us, "depth_device_us": depth_us,
                "color_system_us": color_system_us,
                "depth_system_us": depth_system_us,
                "rgbd_skew_ms": abs(color_us - depth_us) / 1000 if color_us > 0 and depth_us > 0 else None,
                "timestamp_valid": color_us > 0 and depth_us > 0,
                "wait_start_monotonic_ns": start, "received_monotonic_ns": received}

    def close(self):
        if self.started:
            self.started = False
            self.pipeline.stop()


class MockCamera:
    def __init__(self, cfg):
        pass

    def open(self):
        return {"name": "mock_Gemini336", "serial": "SYNTHETIC", "depth_aligned_to_color": False}

    def capture(self, directory, index):
        # Portable mock assets require no SDK, NumPy or OpenCV.
        base = Path(directory) / f"frame_{index:06d}"
        color = base.with_suffix(".ppm")
        depth = base.with_suffix(".pgm")
        color.write_bytes(b"P6\n4 3\n255\n" + bytes([80, 120, 160]) * 12)
        depth.write_bytes(b"P5\n4 3\n65535\n" + (500).to_bytes(2, "big") * 12)
        now = time.monotonic_ns()
        return {"color_file": color.name, "depth_file": depth.name, "depth_scale_mm": 1,
                "depth_encoding": "PGM big-endian uint16", "invalid_depth_raw": 0,
                "rgbd_skew_ms": 0, "synthetic": True, "received_monotonic_ns": now}

    def close(self):
        pass
