import cv2


class Video:
    def __init__(self):
        self._source = cv2.VideoCapture()
        self._codec = cv2.VideoWriter_fourcc(*'mp4v')
        self._buffer = []
        self.frame_size = (0, 0)

    def play(self, frame_callback=None, *args, **kwargs):
        for frame in self.get_frames_from_buffer():
            if frame_callback:
                frame = frame_callback(frame, *args, **kwargs)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(25)
            if key == 'q' or key == 113:
                break

    def save(self, filepath, fps, is_color=True, frame_callback=None, *args, **kwargs):
        assert len(self._buffer), "No frames to save."
        out = cv2.VideoWriter(filepath, self._codec, fps, self.frame_size, is_color)
        for frame in self.get_frames_from_buffer():
            if frame_callback is not None:
                frame = frame_callback(frame, *args, **kwargs)
            out.write(frame)
        out.release()

    def save_raw(self, filepath, fps, is_color=True, frame_callback=None, *args, **kwargs):
        assert len(self._buffer), "No frames to save."
        out = cv2.VideoWriter(filepath.replace('mp4', 'raw'), 0, fps, self.frame_size, is_color)
        for frame in self.get_frames_from_buffer():
            if frame_callback is not None:
                frame = frame_callback(frame, *args, **kwargs)
            out.write(frame)
        out.release()

    def load(self, filepath):
        self.clear()
        self._source.open(filepath)
        assert self._is_source_open(), "Couldn't load video source (filepath={})".format(filepath)
        self.frame_size = (int(self._source.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self._source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # self._buffer = list(self.get_frames_from_source())

    def load_from_camera(self, camera_id=0, max_frames=1000, frame_callback=None, *args, **kwargs):
        self.clear()
        self._source.open(camera_id)
        assert self._is_source_open(), "Couldn't open camera (id={}) video source".format(camera_id)
        self.frame_size = (int(self._source.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self._source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for frame in self.get_frames_from_source(max_frames=max_frames):
            if frame_callback is not None:
                frame = frame_callback(frame, *args, **kwargs)
            self._buffer.append(frame)
            if not Video.show_frame(frame):
                break

    def _is_source_open(self):
        if not self._source or not self._source.isOpened():
            return False
        return True

    def save_frame_to_buffer(self, frame):
        self._buffer.append(frame)
        return frame

    def get_frames_from_buffer(self, skip_frames=0, max_frames=-1):
        if max_frames == -1:
            max_frames = float('inf')
        frame_count = 0
        for frame in self._buffer[skip_frames:]:
            if frame_count > max_frames:
                break
            frame_count += 1
            yield frame

    def get_frames_from_source(self, skip_frames=0, max_frames=-1):
        if max_frames == -1:
            max_frames = float('inf')
        frame_count = 0
        while True:
            is_valid_frame, frame = self._source.read()
            if not is_valid_frame or frame_count >= max_frames + skip_frames:
                break
            frame_count += 1
            if frame_count > skip_frames:
                yield frame

    @staticmethod
    def show_frame(frame, wait_time=25):
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(wait_time)
        if key == 'q' or key == 113:
            return False
        return True

    def clear(self):
        if self._source is not None:
            self._source.release()
        self._buffer = []
        self.frame_size = (0, 0)

    def get_fps(self):
        if self._is_source_open():
            return self._source.get(cv2.CAP_PROP_FPS)
        return None

    def len_buffer(self):
        return len(self._buffer)
