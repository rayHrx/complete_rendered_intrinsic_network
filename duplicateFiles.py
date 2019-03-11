from shutil import copyfile
for i in range(0,63):
    copyfile("dataset/output/face_train/normal.png", "dataset/output/face_train/" + str(i) + "_normal.png")
for i in range(0,63):
    copyfile("dataset/output/face_train/albedo.png", "dataset/output/face_train/" + str(i) + "_depth.png")
    copyfile("dataset/output/face_train/albedo.png", "dataset/output/face_train/" + str(i) + "_albedo.png")
    copyfile("dataset/output/face_train/mask.png", "dataset/output/face_train/" + str(i) + "_mask.png")