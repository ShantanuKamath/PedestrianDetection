# HumanDetection
*- Developed as research project at BMW@NTU Future Mobility Research Lab.*

The application is a native C++ application that makes use of OpenCV Image processing library to process images, detect and classify pedestrians in real driving situations.  
The algorithm developed was implemented from scratch with the adoption of [Histograms of Oriented Gradients](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) (HOG) descriptors and [LibLinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) library for Linear Support Vector Machines (L-SVM) classifiers.  
[KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php) was used to obtain a sufficiently large image dataset.  
This application was used again with Android NDK to use it in a native android application.  

# Results 
![Singular](https://github.com/ShantanuKamath/HumanDetection/Images/singles.jpg)
![Large Scene](https://github.com/ShantanuKamath/HumanDetection/Images/scene.jpg)
