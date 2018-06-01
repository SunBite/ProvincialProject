# -*- coding: utf-8 -*-
classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling","swing", "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
with open("/home/sunbite/videopath.txt",'a') as f:
    for i in classname:
        libsvmfilepath = "/home/sunbite/video/action_youtube_naudio/"+ i+" "
        f.writelines(libsvmfilepath)