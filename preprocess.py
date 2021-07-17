import cv2,os,glob,random,shutil
import numpy as np
from sklearn.model_selection import train_test_split

def merge_frames(frame_list):
    stack = np.stack(tuple(frame_list), axis=2)
    return stack
def save_video(video_ndarray,save_path):
    # 保存npy文件
    np.save(save_path+'.npy', video_ndarray)
    # 保存mp4文件
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') 
    out = cv2.VideoWriter(save_path+'.mp4', fourcc, 8, (256, 256),isColor=False) 
    for i in range(video_ndarray.shape[2]):
        out.write(video_ndarray[:,:,i])  # 写入帧
    

def video_preprocess():
    root = 'data/origin'
    class_folder = os.listdir(root)
    for class_name in class_folder:
        videos_path = glob.glob(os.path.join(root,class_name,'*.mp4')) 
        save_index = 0
        for video_path in videos_path:
            cap=cv2.VideoCapture(video_path)
            print(video_path)
            select_frames = []

            index = 0   # 视频降采样，每3帧里抽一帧
            for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ok,frame = cap.read()
                if index % 3 == 0:
                    frame = cv2.cvtColor(cv2.resize(frame,(256,256)),cv2.COLOR_BGR2GRAY)
                    select_frames.append(frame)
                index += 1

            num_full = len(select_frames)//16
            num_left = len(select_frames)%16

            res = []
            for i in range(num_full):
                res.append(merge_frames(select_frames[i:i+16]))
            if num_left>=8:
                res.append(merge_frames(select_frames[-16:]))


            saved_folder = os.path.join('data/preprocess',class_name)
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
            for video in res:
                save_video(video,os.path.join(saved_folder,class_name+'_'+str(save_index)))
                save_index += 1


def data_split(rate=0.2):
    root = 'data/preprocess'
    train_dir = 'data/train'
    test_dir = 'data/test'
    class_folder = os.listdir(root)
    for class_name in class_folder:
        datas = glob.glob(os.path.join(root,class_name,'*.npy')) 
        train_files,test_files = train_test_split(datas,test_size = rate)
        for file in train_files:
            shutil.copy(file,train_dir)
        for file in test_files:
            shutil.copy(file,test_dir)
if __name__=='__main__':
    video_preprocess()
    data_split()
