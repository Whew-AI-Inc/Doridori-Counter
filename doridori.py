import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.signal import find_peaks
from celluloid import Camera
from tqdm import tqdm

class Doridori:
    def __init__(self,filepath):
        self.cap = cv2.VideoCapture(filepath)
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.df = np.array([])
        self.distance_list = np.array([])
        self.peaks = np.array([])
        self.face_length = np.array([])
        self.filepath = filepath
        
    def detect_face(self):
        frame_cnt = 0
        nose_x = list()
        nose_y = list()
        nose_z = list()
        face_length = list()
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
            while(self.cap.isOpened()):
                ret, frame = self.cap.read()
                if ret:
                    frame_cnt += 1
                    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        x, y, z = self.__getMark(results.multi_face_landmarks)
                        x_, y_, z_ = self.__getMark(results.multi_face_landmarks, mark_num=8)
                        nose_x.append(x)
                        nose_y.append(y)
                        nose_z.append(z)
                        face_length.append(distance.euclidean([x, y, z], [x_, y_, z_]))
                    if frame_cnt >= self.total_frame:
                        print("============End Video============")
                        self.df = np.array([nose_x, nose_y, nose_z]).T
                        self.face_length = np.array(face_length)
                        break
            self.cap.release()
            cv2.destroyAllWindows()
        return self.df

    def fit(self, data = np.array([]), threshold=2.3, min_peak_time = 400, display_mode = True):
        distance_list = list()
        if data.size == 0:
            df = self.df
        else:
            df = data
        for i in range(1, len(df)):
            distance_list.append(distance.euclidean(df[i-1,:], df[i,:]))
        distance_list = self.__normalize(np.array(distance_list))
        min_peak_distance = int(min_peak_time * self.frame_rate / 1000.0)
        peaks_index = find_peaks(distance_list, distance=min_peak_distance)[0]
        low_peak_index = list()
        for i, j in enumerate (peaks_index):
            if distance_list[j] < threshold:
                low_peak_index.append(i)
        peaks_index= np.delete(peaks_index, low_peak_index)
        print(f"total_doridori_count : {len(peaks_index)}")
        peaks = list()
        for i, value in enumerate (distance_list):
            if i in peaks_index:
                peaks.append(value)
            else:
                peaks.append(np.nan)
        if display_mode:
            plt.figure(figsize=(25,8))
            plt.plot(distance_list)
            plt.plot(peaks, 'ro')
            
        self.distance_list = distance_list
        self.peaks = peaks
        
        return len(peaks_index)
    
    def save_video(self, filepath, display_frame = 100, video_size=(25,8)):
        fig, ax = plt.subplots(figsize=video_size)
        camera = Camera(fig)
        padding_nan = np.empty(display_frame)
        padding_nan[:] = np.nan
        distance_with_nan = np.concatenate([padding_nan, self.distance_list])
        peaks_with_nan = np.concatenate([padding_nan, self.peaks])
        for i in tqdm(range(display_frame, len(distance_with_nan))):
            ax.plot(distance_with_nan[i-display_frame:i], c='blue')
            ax.plot(peaks_with_nan[i-display_frame:i], 'ro')
            camera.snap()
        print(f"saving to {filepath}")
        animation = camera.animate(interval=1000.0/self.frame_rate)
        animation.save(filepath)
        plt.close(fig)
        
    def __getMark(self, landmarks, mark_num = 4):
        x = 0
        y = 0
        z = 0
        landmark = list(landmarks)
        mark = landmark[0]
        x = mark.landmark[mark_num].x
        y = mark.landmark[mark_num].y
        z = mark.landmark[mark_num].z
        return x, y, z
    
    def __normalize(self, distance_list):
        return distance_list / self.face_length[1:] * self.frame_rate
    