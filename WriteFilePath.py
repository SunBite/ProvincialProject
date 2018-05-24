# -*- coding: utf-8 -*-
classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling","swing", "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
with open("/home/sunbite/filepath-122.txt",'a') as f:
    for i in classname:
        libsvmfilepath = "/home/sunbite/libsvmfile-122/"+ i +  "/part-00000 "
        f.writelines(libsvmfilepath)