# -*- coding: utf-8 -*-

# import os
import shutil
import svmutil
#
# videopath = "/home/sunbite/video1/action_youtube_naudio_orignal"
# savepath = "/home/sunbite/video1/action_youtube_naudio"
# if os.path.isdir(videopath):
#     # 遍历文件夹
#     for dirpath, dirnames, filenames in os.walk(videopath):
#         for filename in filenames:
#             #filepathandname = os.path.join(dirpath, filename)
#             #print(filepathandname)
#             #os.path.split(filepathandname)
#             print(dirpath)
#             print(filename)
for i in range(1,100):
    for j in range(1,100):
        candg = "-s 0 -t 2 -c "+str(i)+" -g "+str(j) +" -b 1"
        #candg = "-s 0 -t 2 -c " + "11" + " -g " + "9" + " -b 1"
        train_y,train_x = svmutil.svm_read_problem("/home/sunbite/train64-122")
        test_y,test_x = svmutil.svm_read_problem("/home/sunbite/test64-122")
        m = svmutil.svm_train(train_y, train_x, candg)
        predict_label, accuary, prob_estimates = svmutil.svm_predict(test_y, test_x, m,"-b 1")
        with open("/home/sunbite/6464-122accuary", 'a') as f:

            libsvmfilepath = "c:"+str(i)+"    g:"+str(j)+"   accuary:"+str(accuary)+"\n"
            f.writelines(libsvmfilepath)