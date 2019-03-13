#Importing the libraries

import torch #To utilize the dynamic graphs of pytorch which will enable us to very efficiently compute the gradients of composition functions.

from torch.autograd import Variable

import cv2 #To draw rectangles around the object (dog)...not specifically for object detection

from data import BaseTransform , VOC_CLASSES as labelmap
#BaseTransform will help do transformation on the images
#VOC_CLASSES is a disctionary that will do the encoding of the classes

from ssd import build_ssd #Constructor for SSD network

import imageio #Library to process the images of the video

# Defining a function that will do the detections

#Performing frame by frame detection

def detect (frame , net , transform):
    height , weight = frame.shape[:2] #height->0 width-> 1
    #Transformations to fit the input frame into the neural network
    frame_t = transform(frame)[0] # To get the first element of the transform(frame)
    #frame_t is a numpy array
    #We need to convert this numpy array into torch tensor
    x = torch.from_numpy(frame_t).permute(2,0,1) # since green which was at index 2 has to be put as index 0 and Blue which had index 1 has to be put at index 2 so the new sequence is (2,0,1)
    #For SSD we need to convert RGB(0,1,2)to GRB (2,0,1)

    # Now we need to add a fake dimension corresponding to the batch because the neural network cannot actually accept single inputs like single input vetor or a single image...it only accepts it into some batches
    #So we need to create a structure with first input corresponding to the batch and the rest inputs corresponding to the inputs
    x = Variable(x.unsqueeze(0)) #to keep the fake dimension at the zeroth index # We need to convert the torch tensor of inputs into a torch variable
    #Torch variable is a highly advanced variablr which contains both tensor and a gradient

    y = net(x)

    detections =  y.data
    #creating a tensor scale
    scale = torch.tensor([width , height , width , height])

    #detections = [batch , number of classes , number of occurences of the class , (score , x0 , Y0 , x1 , Y1 )] #cooredinates of upper left (x0 , Y1) and lower right corner (x1 , Y1)

    for i in range (detections.size(1)):
        j = 0 # i-> class j-> occurence
        while detections [0,i,j,0] >=0.6 :#0-> batch i , j 0-> index of score
            pt = (detections [0,i,j,1:]*scale).numpy() # "1:" to take the coordinates ->x0 , Y0 , x1 , Y1 # we multiply it with the scale so that the coordinates are normalized and obtained in the scale of the image
            # we need to convert it back to numpy array becuase the rectangle function work only with numpy arrays and not torch tensors
            cv2.rectangle(frame , (int (pt[0]) , int(pt[1])) , (int(pt[2]) , int(pt[3])), (255,0,0),2) #coordinates
            #printing the labels
            cv2.putText(frame , labelmap[i-1] , cv2.FONT_HERSHEY_SIMPLEX , 2 , (255,255,255) , 2 , cv2.LINE_AA)
            j += 1
    return frame

# creating the SSD neural network
net = build_ssd ('test') # to test the ssd on model
# now we load the weigths of an already pre-trained SSD neural network
net.load_state_dict (torch.load('ssd300_mAP_77.43_v2.pth' , map_location = lambda storage , loc : storage))

#creating the transformation
transform = BaseTransform(net.size , (104/256.0 , 117/256.0 , 123/256.0))

#Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps'] # fps-> frames per second
#creating an object that will contain the video
writer = imageio.get_writer('output.mp4' , fps = fps)

for i , frame in enumerate (reader):
    frame = detect (frame , net.eval() , transform)
    writer.append_data(frame)
    print (i) # To see the number of the processed frame

writer.close()


