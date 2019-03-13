# Object-Detection_SSD
Learning attempt on object detection using Single Shot Multi Box Detection.

SSD will divide the image into segments. Make boxes around these segments.Each box individually predicts whether the object is present in it or not.
Also predicts where the object is present in the image.

 Faster R-CNN uses a region proposal network to create boundary boxes and utilizes those boxes to classify objects. 
 While it is considered the state-of-the-art in accuracy, the whole process runs at 7 frames per second.Far below what a real-time processing needs. 
 
 SSD speeds up the process by eliminating the need of the region proposal network.
 
 To recover the drop in accuracy, SSD applies a few improvements including multi-scale features and default boxes. 
 
 These improvements allow SSD to match the Faster R-CNN’s accuracy using lower resolution images, which further pushes the speed higher.
