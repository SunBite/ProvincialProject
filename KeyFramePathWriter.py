# -*- coding: utf-8 -*-
import os
classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling","swing", "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
keyframe = "/home/sunbite/Co_KNN_SVM_TMP/keyframe"
if os.path.isdir(keyframe):
    # 遍历文件夹
    for dirpath, dirnames, filenames in os.walk(keyframe):
        for filename in filenames:
            filepathandname = os.path.join(dirpath, filename)
            print(filepathandname)
            with open("/home/sunbite/Co_KNN_SVM_TMP/keyframepath.txt",'a') as f:
                    f.writelines(filepathandname+"\n")
                    f.close()