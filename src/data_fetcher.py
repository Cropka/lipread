import os
import cv2
from glob import glob
from video import Video


fps = 25
frames_in_batch = 12
frame_size = (24, 32)


class DataFetcher(object):
    def __init__(self):
        self._mouth_ready_files = []

    def add_to_file_list(self, main_dir):
        dir_list = sorted(os.listdir(main_dir))
        for subdir in dir_list:
            self._mouth_ready_files += glob(os.path.join(main_dir, subdir, '*_mouth*'))

    def get_files_containing_words(self, words):
        word_files = {k:[] for k in words}
        for mouth_file in self._mouth_ready_files:
            label_file = mouth_file.replace('_mouth.mp4', '.txt')
            with open(label_file, 'r') as label:
                start_marker = False
                for i, line in enumerate(label):
                    line_parts = line.strip().split(' ')
                    if start_marker:
                        if len(line_parts[0]) and line_parts[0] in words:
                            word_files[line_parts[0]].append((mouth_file, line_parts[1], line_parts[2], str(i+1)))
                    if len(line_parts) == 4 and line_parts[0] == 'WORD' and line_parts[1] == 'START':
                        start_marker = True
        return word_files

    @staticmethod
    def save_samples(word_files, dest_dir, max_files=float('inf')):
        for word, files_info in word_files.items():
            os.makedirs(os.path.join(dest_dir, word), exist_ok=True)
            file_count = 0
            for file_info in files_info:
                whole_vid = Video()
                whole_vid.load(file_info[0])
                if whole_vid.frame_size != frame_size:
                    # print("Skipped due to wrong frame size")
                    continue
                if file_count >= max_files:
                    print("Max files reached for word: {}".format(word))
                    break
                file_count += 1
                start_frame = int(float(file_info[1]) * fps)
                stop_frame = int(float(file_info[2]) * fps)
                frame_span = stop_frame - start_frame
                frame_diff = frames_in_batch - frame_span
                word_vid = Video()
                word_vid.frame_size = frame_size
                frames_to_save = []
                if frame_diff < 0:
                    dropout = int(frame_span / (frame_diff * -1))
                # print("{} {}".format(file_info[0], file_info[3]))
                # print('{} {} {} {}'.format(start_frame, stop_frame, frame_span, frame_diff))
                vid_frames = whole_vid.get_frames_from_source(skip_frames=start_frame, max_frames=frame_span)
                for i, frame in enumerate(vid_frames):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(frames_to_save) < frame_diff:
                        frames_to_save.append(frame)
                    if frame_diff < 0:
                        if (i + 1) % dropout:
                            word_vid.save_frame_to_buffer(frame)
                        else:
                            frame_diff += 1
                            if frame_diff:
                                dropout = int((frame_span - i - 1) / (frame_diff * -1))
                    else:
                        word_vid.save_frame_to_buffer(frame)
                if len(frames_to_save):
                    while word_vid.len_buffer() != frames_in_batch:
                        for frame in frames_to_save:
                            word_vid.save_frame_to_buffer(frame)
                            if word_vid.len_buffer() == frames_in_batch:
                                break
                if word_vid.len_buffer() != frames_in_batch:
                    print("Word vid has wrong number of frames in buffer; {} != {}".format(word_vid.len_buffer(),
                                                                                           frames_in_batch))
                    print("{} {}".format(file_info[0], file_info[3]))
                    print('{} {} {} {}'.format(start_frame, stop_frame, frame_span, frame_diff))
                    continue
                new_file_name = '_'.join(file_info[0].split('/')[-2:]).replace('mouth', file_info[3])
                # frames = word_vid.get_frames_from_buffer()
                # for frame in frames:
                #     Video.show_frame(frame, 2000)
                word_vid.save(filepath=os.path.join(dest_dir, word, new_file_name), fps=fps, is_color=False)


df = DataFetcher()
df.add_to_file_list('lrs3/pretrain')
files = df.get_files_containing_words(['DIFFERENT', 'GOING'])
for w, k in files.items():
    print('{} {}'.format(w, len(k)))
df.save_samples(files, 'word_data')
