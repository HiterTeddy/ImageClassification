import os
import cv2

if __name__ == "__main__":
    image = cv2.imread("data/FatheroftheBridePartIIframe/FatheroftheBridePartII_i_frame_1860.jpg")
    print(image.shape)
    
    # root = "data"
    # file_to_write = open(os.path.join(root, "files.txt"), "w")
    
    # with open(os.path.join(root, "classes.txt")) as file:
    #     dirs = file.readlines()
        
    #     for i in range(len(dirs)):
    #         dir = dirs[i]
    #         dir_path = os.path.join(root, dir[0:-1])
    #         files = os.listdir(dir_path)
    #         for image in files:
    #             file_to_write.write(str(os.path.join(dir_path, image)) + "," + str(i) + "\n")
    
    # file_to_write.close()
            