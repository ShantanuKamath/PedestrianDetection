//
//  m.cpp
//  OpenCVTest
//
//  Created by Shantanu Kamath on 12/12/15.
//  Copyright © 2015 Shantanu Kamath. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

int main1( int argc, char** argv )
{
    string imageName("1.jpg");
    if( argc > 1)
    {
        imageName = argv[1];
    }

    Mat image;
    image = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
    
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    
    
    
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}