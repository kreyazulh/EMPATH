import cv2
import os

#change to your path
cam = cv2.VideoCapture("/home/Desktop/Thesis/Research Codebase/dance.mp4") #specify VideoCapture
fps = cam.get(cv2.CAP_PROP_FPS)
output_folder = "/home/Desktop/Thesis/Research Codebase/dance"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(fps)

n = 0
i = 0
frames = 4  #vary this to change the number of frames to be extracted
while True:
    ret, frame = cam.read()
    if(frames*n)%8==0:  #8 is hardcoded here
        frame_path = os.path.join(output_folder, "{}.jpg".format(i))
        cv2.imwrite(frame_path, frame)
        i+=1
    n+=1
    if ret == False:
        break
cam.release()
cv2.destroyAllWindows()
