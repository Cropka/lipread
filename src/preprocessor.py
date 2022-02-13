import os
import sys
import logging
import subprocess
import shutil
import cv2
import numpy as np
import yt_dlp as youtube_dl
from video import Video


logger = logging.getLogger('preprocessor_logger')
cout = logging.StreamHandler()
cout.setLevel(logging.DEBUG)
cout.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log_file = logging.FileHandler('preprocessor.log')
log_file.setLevel(logging.DEBUG)
log_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(cout)
logger.addHandler(log_file)
logger.setLevel(logging.DEBUG)


class Preprocessor(object):
    def __init__(self, target_shape, ydl_opts, fps):
        self.target_shape = target_shape
        self._yt_dl_opts = ydl_opts
        self._fps = fps
        self._tracker = cv2.TrackerCSRT_create()
        self._face_detector = cv2.CascadeClassifier(
            'venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        self._mouth_detector = cv2.CascadeClassifier(
            'venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_mouth.xml')

    def preprocess_dir(self, main_dir, start_ind=0, end_ind=None):
        dir_list = sorted(os.listdir(main_dir))
        if end_ind is None:
            end_ind = len(dir_list)
        dir_list = dir_list[start_ind:end_ind]
        logger.info("Preprocessing initiated on '{}'".format(main_dir))
        for i, subdir in enumerate(dir_list):
            current_vid_dir = os.path.join(main_dir, subdir)
            current_vid_id = self._get_video_id(current_vid_dir)
            logger.info("Start preprocessing of {} th video (id = {})".format(i, current_vid_id))
            is_downloaded = self.safe_download_yt_video(current_vid_id)
            if not is_downloaded:
                shutil.rmtree(current_vid_dir)
            original_filepath = self._yt_dl_opts['outtmpl'] % {'id': current_vid_id, 'ext': self._yt_dl_opts['format']}
            logger.info("Downloaded {} th video - {}".format(i, original_filepath))
            self.preprocess_video(original_filepath, current_vid_dir)
            os.unlink(original_filepath)

    def preprocess_video(self, video_filepath, label_dir):
        original_vid = self._load_video(video_filepath)
        frame_cursor = 0
        for label_file in sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')]):
            logger.info("Working on part: {}".format(label_file))
            frame_labels = self._parse_label_file(label_dir, label_file)
            skip_frames = frame_labels[0][0] - 1 - frame_cursor
            if skip_frames < 0:
                skip_frames = 0
            frame_cursor = frame_labels[-1][0]
            part_vid = Video()
            part_vid.frame_size = self.target_shape
            frames = original_vid.get_frames_from_source(skip_frames=skip_frames, max_frames=len(frame_labels) - 1)
            processed_frames = self._process_frames(frames, frame_labels)
            # for frame in processed_frames:
            #     part_vid.save_frame_to_buffer(frame)
            # part_vid.play()
            # partname_root = os.path.splitext(label_file)[0]
            # output_vid_filepath = os.path.join(label_dir, '{}_mouth.mp4'.format(partname_root))
            # part_vid.save(output_vid_filepath, self._fps, False)

    @staticmethod
    def _get_video_id(subdir):
        for file in os.listdir(subdir):
            if file.endswith(".txt"):
                full_file = os.path.join(subdir, file)
                with open(full_file, 'r') as label_file:
                    for i, line in enumerate(label_file):
                        if i == 2:
                            return line.split(':')[1].strip()

    def safe_download_yt_video(self, vid_id):
        for try_nr in range(10):
            try:
                self._download_yt_video(vid_id)
            except youtube_dl.DownloadError as e:
                if "Private video." in str(e):
                    logger.error('Video {} is private. Removing label directory...'.format(vid_id))
                    return False
                elif "Video unavailable" in str(e):
                    logger.error(
                        "Video {} is not available anymore. Removing label directory...".format(vid_id))
                    return False
                else:
                    logger.error(
                        "Error occurred during {}th try of downloading the {} video".format(try_nr, vid_id),
                        exc_info=True)
            else:
                return True

        logging.error("Downloading of the {} video could not be performed".format(vid_id))
        raise youtube_dl.DownloadError("Custom msg: Downloading of the {} video could not be performed"
                                       .format(vid_id))

    def _download_yt_video(self, vid_id):
        vid_url = "https://youtu.be/{}".format(vid_id)
        with youtube_dl.YoutubeDL(self._yt_dl_opts) as ydl:
            ydl.download([vid_url])

    def _load_video(self, filepath):
        vid = Video()
        vid.load(filepath)
        vid_fps = vid.get_fps()
        if not self._fps - 0.01 < vid_fps < self._fps + 0.01:
            logger.info("Video has fps = {}. Converting to {} fps...".format(vid_fps, self._fps))
            self._change_fps_of_video(filepath, self._fps)
            logger.info("Video has been converted.")
            vid.clear()
            vid.load(filepath)
        return vid

    @staticmethod
    def _change_fps_of_video(filepath, fps):
        output_root, output_ext = os.path.splitext(filepath)
        output_file = '{}_changed_fps{}'.format(output_root, output_ext)
        subprocess.run(['ffmpeg', '-y', '-i', filepath, '-r', str(fps), output_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.rename(output_file, filepath)

    @staticmethod
    def _parse_label_file(label_dir, label_filename):
        with open(os.path.join(label_dir, label_filename), 'r') as label:
            frame_labels = []
            empty_line_encountered = False
            for line in label:
                if not line.strip():
                    if not empty_line_encountered:
                        empty_line_encountered = True
                    else:
                        break
                if line[0].isdigit():
                    frame_labels.append([float(v) for v in line.split(' \t')])
        assert frame_labels, "Broken label text file: {}".format(os.path.join(label_dir, label_filename))
        return frame_labels

    def _process_frames(self, frames, frame_labels):
        processed_frames = []
        is_tracker_initialized = False
        last_mouth_rect = (0, 0, 0, 0)
        for frame_ind, frame in enumerate(frames):
            x, y, w, h = Preprocessor.convert_relative_rect_format_to_normal(frame.shape, *frame_labels[frame_ind][1:])
            try:
                # mask = np.zeros(frame.shape[:2], dtype='uint8')

                cropped_face_frame = frame[y:y+h, x:x+w]                     # face region
                # mouth_mask = cv2.rectangle(mask, (x, y + int(0.5 * h)), (x+w, int(y + h)), 255, -1)
                cropped_mouth_frame = frame[y + int(0.5 * h):int(y + h), x:x + w]  # mouth region
                # masked_frame = cv2.bitwise_and(frame, frame, mask=mouth_mask)
                processing_frame = cropped_mouth_frame
                # print("cropped frame shape = {}".format(processing_frame.shape))
                if processing_frame.shape[0] <= last_mouth_rect[1] + last_mouth_rect[3] or processing_frame.shape[1] <= last_mouth_rect[0] + last_mouth_rect[2]:
                    is_tracker_initialized = False
                m_x, m_y, m_w, m_h = self.get_mouth_rect(processing_frame)
                if m_x is None:
                    if not is_tracker_initialized:
                        if not last_mouth_rect[2] or not last_mouth_rect[3]:
                            m_x, m_y, m_w, m_h = (0, 0, processing_frame.shape[1], processing_frame.shape[0])
                        else:
                            m_x, m_y, m_w, m_h = last_mouth_rect
                    else:
                        try:
                            bbox = self._update_tracker(processing_frame)
                            if bbox[0] is not None:
                                m_x, m_y, m_w, m_h = [int(i) for i in bbox]
                            else:
                                m_x, m_y, m_w, m_h = last_mouth_rect
                        except cv2.error as e:
                            if not last_mouth_rect[2] or not last_mouth_rect[3]:
                                m_x, m_y, m_w, m_h = (0, 0, processing_frame.shape[1], processing_frame.shape[0])
                            else:
                                m_x, m_y, m_w, m_h = last_mouth_rect
                else:
                    self._tracker.init(processing_frame, (int(m_x), int(m_y), int(m_w), int(m_h)))
                    #logger.debug("Tracker initiated with frame of shape = {} and ROI = {}, {}, {}, {}".format(processing_frame.shape, int(m_x), int(m_y), int(m_w), int(m_h)))
                    is_tracker_initialized = True

                mouth_frame = processing_frame[m_y:m_y + m_h, m_x:m_x + m_w]
                if not mouth_frame.shape[0] or not mouth_frame.shape[1]:
                    logger.error("Error: Mouth frame empty!")
                    logger.error("Mouth rect: {}, {}, {}, {}".format(m_x, m_y, m_w, m_h))
                    mouth_frame = processing_frame

#                drawed_frame = draw_rect(processing_frame, m_x, m_y, m_w, m_h)
                resized_frame = Preprocessor._resize_frame(mouth_frame, self.target_shape)
                grayed_frame = convert_to_grayscale(resized_frame)
                last_mouth_rect = (m_x, m_y, m_w, m_h)
                Video.show_frame(frame, 15000)
                Video.show_frame(cropped_face_frame, 15000)
                Video.show_frame(mouth_frame, 15000)
                Video.show_frame(grayed_frame, 15000)
            except cv2.error as e:
                logger.error("Error during frame processing pipeline!", exc_info=True)
                logger.error('Shape of cropped mouth frame -- {}'.format(cropped_mouth_frame.shape))
                logger.error('Shape of mouth frame -- {}, mouth rect -- {}, {}, {}, {}'
                             .format(mouth_frame.shape, m_x, m_y, m_w, m_h))
                Video.show_frame(processing_frame, 10000)
                raise
            processed_frames.append(grayed_frame)
        return processed_frames

    @staticmethod
    def convert_relative_rect_format_to_normal(frame_size, x, y, w, h):
        x = frame_size[1] * x if not x < 0 else 0
        y = frame_size[0] * y if not y < 0 else 0
        w = frame_size[1] * w if not w < 0 else 0
        h = frame_size[0] * h if not h < 0 else 0
        x = frame_size[1] if x > frame_size[1] else x
        y = frame_size[0] if y > frame_size[0] else y
        w = frame_size[1] - x if x + w > frame_size[1] else w
        h = frame_size[0] - y if y + h > frame_size[0] else h
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def normalize_rect(frame_size, x, y, w, h):
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x = frame_size[1] if x > frame_size[1] else x
        y = frame_size[0] if y > frame_size[0] else y
        w = frame_size[1] - x if x + w > frame_size[1] else w
        h = frame_size[0] - y if y + h > frame_size[0] else h
        return int(x), int(y), int(w), int(h)

    def get_mouth_rect(self, frame):
        converted = convert_to_grayscale(frame)
        min_height = int(frame.shape[0] * 0.35)
        min_width = int(frame.shape[1] * 0.35)
        try:
            mouths_rect = self._mouth_detector.detectMultiScale(converted, minNeighbors=3, scaleFactor=1.1,
                                                                minSize=(min_height, min_width))
        except cv2.error as e:
            print(e)
            return
        if not len(mouths_rect):
            return None, None, None, None
        return mouths_rect[0]

    def get_face_rect(self, frame):
        converted = convert_to_grayscale(frame)
        min_height = int(frame.shape[0] * 0.1)
        min_width = int(frame.shape[1] * 0.1)
        try:
            faces_rect = self._face_detector.detectMultiScale(converted, minNeighbors=3, scaleFactor=1.15,
                                                              minSize=(min_height, min_width))
        except cv2.error as e:
            print(e)
            return
        if not len(faces_rect):
            return None
        return faces_rect[0]

    def _update_tracker(self, frame):
        try:
            status_tracker, bbox = self._tracker.update(frame)
        except cv2.error as e:
            logger.error("Error during tracker update!", exc_info=True)
            logger.error("Shape of frame passed to tracker: {}".format(frame.shape))
            raise
        if status_tracker:
            return Preprocessor.normalize_rect(frame.shape, *bbox)
        logger.error('Tracker update returned status -- {}'.format(status_tracker))
        return None, None, None, None

    @staticmethod
    def _resize_frame(frame, t_shape):
        try:
            return cv2.resize(frame, t_shape)
        except cv2.error as e:
            logger.error("Error during frame resizing!", exc_info=True)
            logger.error("Shape of frame to be resized: {}!".format(frame.shape))
            raise


def set_log_level(log_level):
    logger.setLevel(log_level)


def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def draw_rect(frame, x, y, w, h, c_blue=255, c_green=0, c_red=0):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (c_blue, c_green, c_red), 2)
    return frame


ydl_opts = {
    'outtmpl': os.path.join('spam', '%(id)s.%(ext)s'),
    'format': 'mp4',
    'cachedir': False
}
target_shape = (24, 32)


p = Preprocessor(target_shape=target_shape, ydl_opts=ydl_opts, fps=25)
p.preprocess_dir(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
