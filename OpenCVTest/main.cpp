#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <queue>
#include <string>
#include "opencv2/ml.hpp"
#include "../LibLinear/linear.h"
#include "../LibLinear/linear.cpp"
using namespace cv;
using namespace std;
using namespace cv::ml;

// DEFINING CONSTANTS
#define BINS 9                  //No. of columns in the histogram.
#define CELL_SIZE 8             // Cellsize px x cellsize px form one cell.
#define PATCH_WIDTH 64
#define PATCH_HEIGHT 128        // 64 X 128 IMAGE PATCH
#define ROWS_CELLTABLE PATCH_HEIGHT/CELL_SIZE //No. of rows in a cell table eg. 128 / 8 = 16
#define COLS_CELLTABLE PATCH_WIDTH/CELL_SIZE //No. of cols in a cell table ex. 64 / 8 = 8
#define SHIFT_PATCH 8 // SHIFT THE PATCH IN X AND Y BY ___ pixels
#define RESIZE_PATCH 0.80 // 80% of the original image
#define BLOCK_SIZE 2 // _ x _ no. of cell in a block

struct rectImg{
    Rect r;
    double decision;
    double rescale;
    bool operator<(const rectImg &o) const
    {
        return (decision< o.decision);
    }
};
priority_queue<rectImg> pq;
problem training;
feature_node *testing;
int label;
vector<double> yy;
void writeSVM(vector<vector<vector<double>>> feature)
{
    int x=1;
    int count=0;
    for (int i = 0; i < feature.size(); i++)
    {
        for (int j = 0; j < feature[i].size(); j++)
        {
            for (int k=0; k< feature[i][j].size();k++)
            {
                double num=feature[i][j][k];
                if(num!=0 && !isnan(num))
                {
                    count++;
                }
                x++;
            }
        }
    }
    x=1;
    yy.push_back(label);
    training.x[yy.size()-1] = (feature_node*) malloc(sizeof *training.x[0] * (count+1)); // strictly speaking, the sizeof
    count=0;
    for (int i = 0; i < feature.size(); i++)
    {
        for (int j = 0; j < feature[i].size(); j++)
        {
            for (int k=0; k< feature[i][j].size();k++)
            {
                double num=feature[i][j][k];
                if(num!=0 && !isnan(num))
                {
                    training.x[yy.size()-1][count].value=num;
                    training.x[yy.size()-1][count].index=x;
                    count++;
                }
                x++;
            }
        }
    }
    training.x[yy.size()-1][count].index=-1;
}




void writeFeatureNode(vector<vector<vector<double>>> feature)
{
    int x=1;
    int count=0;
    for (int i = 0; i < feature.size(); i++)
    {
        for (int j = 0; j < feature[i].size(); j++)
        {
            for (int k=0; k< feature[i][j].size();k++)
            {
                double num=feature[i][j][k];
                if(num!=0 && !isnan(num))
                {
                    count++;
                }
                x++;
            }
        }
    }
    x=1;
    testing = (feature_node*) malloc(sizeof *training.x[0] * (count+1)); // strictly speaking, the sizeof
    count=0;
    for (int i = 0; i < feature.size(); i++)
    {
        for (int j = 0; j < feature[i].size(); j++)
        {
            for (int k=0; k< feature[i][j].size();k++)
            {
                double num=feature[i][j][k];
                if(num!=0 && !isnan(num))
                {
                    testing[count].value=num;
                    testing[count].index=x;
                    count++;
                }
                x++;
            }
        }
    }
    testing[count].index=-1;
}








/**
 * TESTING (WRITING TRAINING FILE) PURPOSE.
 * To create test file.
 * Write the mat values in the format needed by predict
 */
void writeFeatureToTextFile(vector<vector<vector<double>>> feature)
{
    int x=1;
    std::ofstream outfile;
    string label ="-1 ";
    string line="";
    
    outfile.open("a1a.t", std::ios_base::app); //Change to test.txt for train purposes.
    // Change to a1a.t for test purposes.
    for (int i = 0; i < feature.size(); i++)
    {
        for (int j = 0; j < feature[i].size(); j++)
        {
            for (int k=0; k< feature[i][j].size();k++)
            {
                double num=feature[i][j][k];
                if(num!=0 && !isnan(num))
                    line+=to_string(x)+":"+to_string(num)+" ";
                x++;
            }
        }
    }
    // cout<<label<<line<<endl;
    
}

/*
 * Function returns a 3D Vector with the values of all cell histograms.
 * The first two dimensions are to navigate throught the cell table.
 * The third dimesion is a vector with the magnitude for the respective bin values
 */
vector<vector<vector<double>>> cellHistograms(Mat magnitude, Mat angle)
{
    vector< vector<vector<double>>> cellHist(ROWS_CELLTABLE,vector<vector<double>>(COLS_CELLTABLE, vector<double>(BINS)));
    int m=-1;
    for(int a=0; a<PATCH_HEIGHT; a+=CELL_SIZE)
    {
        m++;
        int n=-1;
        for (int b=0; b<PATCH_WIDTH; b+=CELL_SIZE)
        {
            Mat magCell=magnitude.rowRange(a, a+CELL_SIZE).colRange(b, b+CELL_SIZE);
            Mat angCell=angle.rowRange(a,a+CELL_SIZE).colRange(b, b+CELL_SIZE);
            n++;
            for (int i=0; i<angCell.rows; i++)
            {
                for (int j=0; j<angCell.cols; j++)
                {
                    float colNumber= angCell.at<float>(i,j);
                    colNumber/=20;
                    colNumber=int(colNumber);
                    
                    if(colNumber==9)                    // When angle is 180. 180/20 is 9 which is a unique case.
                        colNumber=8;
                    cellHist[m][n][colNumber]+=magCell.at<float>(i,j);
                }
            }
        }
    }
    return cellHist;
}





/*
 * Calculates the normalization value and returns it.
 * Method : L2 Normalisation
 */
double normalization(vector<int> blockHist)
{
    double norm=0.0;
    for (int i=0; i<blockHist.size(); i++)
    {
        //blockHist[i]=blockHist[i]/4; not sure why
        norm+=blockHist[i]*blockHist[i];
    }
    
    return sqrt(norm);
}

/*
 * Forms blocks and uses the blocks to normalise all cell histograms within the block.
 * Blocks are overlapping.
 */
void blockNormalisation(vector<vector<vector<double>>> before)
{
    vector<vector<vector<double>>> mean(ROWS_CELLTABLE,vector<vector<double>>(COLS_CELLTABLE, vector<double>(BINS)));
    
    for (int i = 0; i < mean.size(); i++)
    {
        for (int j = 0; j < mean[i].size(); j++)
        {
            for (int k=0; k< mean[i][j].size();k++)
            {
                mean[i][j][k]=0;
            }
        }
    }
    
    vector<vector<vector<double>>> after(ROWS_CELLTABLE,vector<vector<double>>(COLS_CELLTABLE, vector<double>(BINS)));
    copy(before.begin(), before.end(), after.begin());
    
    for (int i =0; i<ROWS_CELLTABLE-1; i++)
    {
        for (int j=0; j<COLS_CELLTABLE-1; j++)
        {
            vector<int> blockHist (BINS);
            for(int a=i; a<i+BLOCK_SIZE; a++)
            {
                for (int b=j; b<j+BLOCK_SIZE; b++)
                {
                    for (int c=0; c<BINS; c++)
                    {
                        blockHist[c]+=before[a][b][c];
                    }
                }
            }
            
            double norm = normalization(blockHist);
            for(int a=i; a<i+BLOCK_SIZE; a++)
            {
                for (int b=j; b<j+BLOCK_SIZE; b++)
                {
                    for (int c=0; c<BINS; c++)
                    {
                        if(after[a][b][c]==before[a][b][c]) // NOT BEEN NORMALISED BEFORE
                        {
                            after[a][b][c]=before[a][b][c]/norm;
                            mean[a][b][c]++;
                        }
                        else                                // HAS BEEN NORMALISED. HAVE TO TAKE MEAN
                        {
                            double temp= after[a][b][c]*mean[a][b][c];
                            mean[a][b][c]++;
                            after[a][b][c]=((before[a][b][c]/norm) + temp)/(mean[a][b][c]);
                        }
                    }
                    
                }
            }
        }
    }
    
    //    // Printing out data to be written. (Final Feature Vector)
    //    int x=1;
    //    for (int i = 0; i < after.size(); i++)
    //    {
    //        for (int j = 0; j < after[i].size(); j++)
    //        {
    //            for (int k=0; k< after[i][j].size();k++)
    //            {
    //                if(after[i][j][k]!=0)
    //                    cout<<x<<":"<<after[i][j][k]<<" ";
    //                x++;
    //            }
    //        }
    //    }
    
    // For training purpose: To create a model
    // writeSVM(after);
    writeFeatureNode(after);
    //writeFeatureToTextFile(after);
    cout<<endl;
}





/*
 * Recieves an image and converts it to gray scale.
 * Calculates gradients in the x and y direction.
 * Using the gradients, it calculates magnitude and direction.
 * It calls cellhistograms function and sends the value for BlockNormalisation.
 */
void calculateGradients( Mat src)
{
    Mat src_gray; // Grayed image
    Mat grad; // Gradient image
    //
    //    const char* window_name = "Gradient calculations";  // Create window
    //    namedWindow( window_name, WINDOW_AUTOSIZE );
    //
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    
    // Convert it to gray
    cvtColor( src, src_gray, COLOR_RGB2GRAY );
    
    // Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat magnitude;
    Mat angle;
    
    // Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    
    // Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    //    /// Total Gradient (approximate)
    //    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    //    cout<<"grad "<<grad<<endl;
    
    cartToPolar(grad_x, grad_y, magnitude, angle, true);
    
    for (int i=0; i<angle.rows; i++) {
        for (int j=0; j<angle.cols; j++) {
            if(angle.at<float>(i,j)>180)
                angle.at<float>(i,j)-=180;
        }
    }
    
    blockNormalisation(cellHistograms(magnitude,angle));
    
    //imshow( window_name, grad );
    //! [wait]
    //waitKey(0); // Wait for a keystroke in the window
    //! [wait]
}


/*
 * Read the file name and returns an image mat file
 */
Mat readImage(string filename)
{
    Mat src; // Source Image
    string imageName(filename);
    src = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
    if( src.empty() )   // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
    }
    return src;
}
//Modified Predict function for decision values.
double predict(const model *model_, const feature_node *x, double &decision)
{
    double *dec_values = Malloc(double, model_->nr_class);
    double label=predict_values(model_, x, dec_values);
    decision=*dec_values;
    free(dec_values);
    return label;
}



/*
 * Scanning through images
 * If image size is greater than the patch. It iterates through the image patch by patch.
 * Also rescales the image to search again.
 */
void patchSelection(Mat src)
{
    int counter1=0;
    int counter2=0;
    int counter3=0;
    int row= src.rows;
    int col= src.cols;
    Mat rect1=src;
    double numrec=0;
    while (src.rows>=PATCH_HEIGHT && src.cols>=PATCH_WIDTH) // While size of picture is greater than 64 x 128
    {
        for (int i=PATCH_HEIGHT; i<=src.rows; i+=SHIFT_PATCH)
        {
            counter2++;
            for (int j=PATCH_WIDTH; j<=src.cols; j+=SHIFT_PATCH)
            {
                counter3++;
                Mat patch =src.rowRange(i-PATCH_HEIGHT, i).colRange(j-PATCH_WIDTH, j);
                calculateGradients(patch);
                model TestOne = *load_model("TrainingSet");
                double decision;
                const char* window_name = "Person Detection Window";  // Create window
                ///......
                
                if(predict(&TestOne, testing, decision)==1.0 )
                {
                    //                    if (counter1<4)
                    //                        continue;
                    //                    namedWindow( window_name, WINDOW_AUTOSIZE );
                    //                    imshow( window_name, patch);
                    //                    waitKey(0); // Wait
                    rectImg RectangleImage;
                    RectangleImage.r=Rect((j-PATCH_WIDTH)/(pow(0.8, counter1)),(i-PATCH_HEIGHT)/(pow(0.8, counter1)),PATCH_WIDTH/(pow(0.8, counter1)),PATCH_HEIGHT/(pow(0.8, counter1)));
                    RectangleImage.decision=decision;
                    RectangleImage.rescale=counter1;
                    pq.push(RectangleImage);
                    //                  Mat orginal=readImage("IMG_4057.JPG");
                    //                  rectangle(orginal, RectangleImage.r, Scalar(26,98,26));
                    //                  namedWindow( window_name, WINDOW_AUTOSIZE );
                    //                  imshow( window_name, orginal);
                    //                  waitKey(0); // Wait
                }
                cout<<predict(&TestOne, testing, decision)<<endl;
                cout<<decision<<endl;
                
            }
        }
        resize(src, src, Size(),RESIZE_PATCH , RESIZE_PATCH);
        counter1++;
        
    }
    cout<<endl;
    cout<<row<<"   "<<col<<endl;
    cout<<counter3<<endl;
    cout<<counter2<<endl;
    cout<<counter1<<endl;
    cout<<numrec<<endl;
}



/*
 * TRAINING PURPOSE.
 * Returns an image in the correct patch size.
 *
 */
void fixDimension(double x1, double y1, double x2, double y2, string filename, int count)
{
    double extraWidth=0;
    double extraHeight=0;
    double width = x2-x1;
    double height = y2-y1;
    if(width <= height/2)
    {
        extraWidth=((height/2)-width)/2;
        width=height/2;
    }
    else
    {
        extraHeight=(width*2-height)/2;
        height=width*2;
    }
    Mat src = readImage("data_object_image_2/training/image_2/"+filename+".png");
    if(x1-extraWidth+width>src.cols)
    {
        double temp=x1-extraWidth+width-src.cols;
        extraWidth+=temp;
    }
    if(x1-extraWidth<0)
    {
        x1=0;
        extraWidth=0;
    }
    if (y1-extraHeight+height>src.rows)
    {
        double temp=y1-extraHeight+height-src.rows;
        extraHeight+=temp;
    }
    if(y1-extraHeight<0)
    {
        y1=0;
        extraHeight=0;
    }
    
    if(x1-extraWidth+width> src.cols)
        width=src.cols-(x1-extraWidth);
    if(y1-extraHeight+height>src.rows)
        height=src.rows-(y1-extraHeight);
    Mat resizedImage;//dst image
    Rect myROI(x1-extraWidth, y1-extraHeight, width, height);
    Mat croppedImage = src(myROI);
    Size size(PATCH_WIDTH,PATCH_HEIGHT);//the dst image size,e.g.100x100
    resize(croppedImage,resizedImage,size);//resize image
    //    patchSelection(resizedImage);
    //    ostringstream convert;   // stream used for the conversion
    //    convert << count;        // insert the textual representation of 'Number' in the characters in the stream
    //    imwrite("NEG/negImg"+convert.str()+".jpg", resizedImage);
    string window_name = ""+filename;
    /// Create window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    imshow( window_name, resizedImage );
    //        imwrite("resized.jpg", resizedImage);
    waitKey(0); // Wait for a keystroke in the window
}





/*
 * TRAINING PURPOSE.
 * Function runs through the label files and detects the word pedestrian.
 * On detection, it gets the coordinates for the rectangle frame in which the pedestrian can be found.
 * This coordinates are then sent to fixDimension to scale it into the needed size.
 */
void searchForPedestrian()
{
    int count=0;
    for (int i=0; i<7481; i++) //Refers to the number of text files or labels.
    {
        
        // Filename making using the loop counter.
        string filename="00";
        if(i<10)
            filename+="000"+to_string(i);
        else if (i<100)
            filename+="00"+to_string(i);
        else if (i<1000)
            filename+="0"+to_string(i);
        else if (i<10000)
            filename+=to_string(i);
        
        // Opening the file.
        ifstream myfile ("training/label_2/"+filename+".txt");
        if (myfile.is_open())
        {
            //Fetching line by line from the file
            string line;
            while ( getline (myfile,line) )
            {
                stringstream ss(line);
                string word;
                
                //Recieve token by token from each line
                while(ss>>word)
                {
                    if (word.compare("Pedestrian")==0)
                    {
                        count++;
                        double x1 , x2 , y1, y2;
                        ss>>x1;
                        ss>>x1;
                        ss>>x1;
                        
                        
                        //Reading tokens into coordinates.
                        ss>>x1;
                        ss>>y1;
                        ss>>x2;
                        ss>>y2;
                        
                        //Getting Image
                        fixDimension(x1,y1,x2,y2,filename, count);
                    }
                    else
                        break;
                }
                cout<<count<<endl;
            }
        }
    }
}

// JUST USED TO RENUMBER IMAGES IN A CONTINUOUS MANNER
void reNumberImages()
{
    int count=1;
    for (int i=0; i<10000; i+=3) //Refers to the number of text files or labels.
    {
        string filename="NEG/negImg"+to_string(i)+".jpg"; // "Selected Pos Images/posImg"
        Mat src=readImage(filename);
        if (!src.empty())
        {
            if (count<=999)
                imwrite("DATA FINAL/TRAIN/NEG/ImgNeg"+to_string(count)+".jpg", src); //"DATA FINAL/TRAIN/POS/ImgPos"
            else
                imwrite("DATA FINAL/TEST/NEG/ImgNeg"+to_string(count-999)+".jpg", src);
            count++;
        }
    }
}


void printPositivePatches(Mat src)
{
    std::ifstream infile("output");
    string line="";
    int y=1;
    while (src.rows>=PATCH_HEIGHT && src.cols>=PATCH_WIDTH)
    {
        for (int i=PATCH_HEIGHT; i<=src.rows; i+=SHIFT_PATCH)
        {
            for (int j=PATCH_WIDTH; j<=src.cols; j+=SHIFT_PATCH)
            {
                getline(infile, line);
                int x=atoi(line.c_str());
                cout<<x<<endl;
                y++;
                
                if(x==1 && y>5000)
                {
                    Mat patch =src.rowRange(i-PATCH_HEIGHT, i).colRange(j-PATCH_WIDTH, j);
                    string window_name = "jgjg";
                    /// Create window
                    namedWindow( window_name, WINDOW_AUTOSIZE );
                    imshow( window_name, patch);
                    waitKey(0); // Wait
                }
            }
        }
        resize(src, src, Size(),RESIZE_PATCH , RESIZE_PATCH);
    }
}




void PositiveImageCalc()
{
    for (int i=1; i<1000; i++) //Refers to the number of text files or labels.
    {
        string filename="DATA FINAL/TRAIN/POS/ImgPos"+to_string(i)+".jpg";
        Mat src=readImage(filename);
        if (!src.empty())
        {
            patchSelection(src);
        }
    }
}



void NegativeImageCalc()
{
    for (int i=1; i<1000; i++) //Refers to the number of text files or labels.
    {
        string filename="DATA FINAL/TRAIN/NEG/ImgNeg"+to_string(i)+".jpg";
        Mat src=readImage(filename);
        if (!src.empty())
        {
            patchSelection(src);
        }
    }
}

void generatedRandomNegatives ()
{
    int x=0;
    for (int i=0; i<7481; i++) //Refers to the number of text files or labels.
    {
        // Filename making using the loop counter.
        string filename="00";
        if(i<10)
            filename+="000"+to_string(i);
        else if (i<100)
            filename+="00"+to_string(i);
        else if (i<1000)
            filename+="0"+to_string(i);
        else if (i<10000)
            filename+=to_string(i);
        
        // Opening the file.
        ifstream myfile ("training/label_2/"+filename+".txt");
        if (myfile.is_open())
        {
            int count=0;
            //Fetching line by line from the file
            string line;
            while ( getline (myfile,line) )
            {
                stringstream ss(line);
                string word;
                
                //Recieve token by token from each line
                while(ss>>word)
                {
                    if (word.compare("Pedestrian")==0)
                    {
                        count++;
                        break;
                    }
                    
                }
            }
            if(count>0)
                break;
            else
            {
                if(x>1000) //Counter to find only 1000 image patches
                    break;
                // CURRENT IMAGE HAS NO PEDESTRIAN
                // FIND PATCH IMAGE TO CHOOSE AS NEGATIVE RANDOMLY
            }
        }
    }
}


/*
 *
 *
 *
 */
void trainAndSaveModel()
{
    training.l=999*2;
    training.n=training.l;
    training.bias=-1;
    training.x=(feature_node**) malloc(sizeof(feature_node*)*training.l);
    training.y = new double[training.l];
    label=1;
    PositiveImageCalc();
    label=-1;
    NegativeImageCalc();
    for(int i=0; i<training.l;i++)
        training.y[i]=yy[i];
    parameter *param = new parameter;
    param->solver_type=L2R_LR;
    param->C=1;
    param->eps=0.01;
    param->nr_weight=0;
    param->init_sol=NULL;
    const char *n=check_parameter(&training, param);
    if (n == NULL) {
        model one = *train(&training, param);
        save_model("TrainingSet", &one);
    }
    
}

void predictWithModel(String filename)
{
    Mat src= readImage(filename);
    patchSelection(src);
    //    model TestOne = *load_model("TrainingSet");
    //    cout<<predict(&TestOne, testing);
    
}
void detectInCompressions()
{
    int x=1;
    int c=1;
    while(c<=10)
    {
        ostringstream convert;   // stream used for the conversion
        convert << c;
        int cc=1;
        while (cc<=5)
        {
            ostringstream ext;   // stream used for the conversion
            switch(cc)
            {
                case 1: ext<<"-0.2";
                    break;
                case 2: ext<<"-0.02";
                    break;
                case 3: ext<<"-0.5";
                    break;
                case 4: ext<<"-1.0";
                    break;
                case 5: ext<<"-1.5";
                    break;
            }
            predictWithModel("CompressedImages/reconstructedImages/output/"+convert.str()+ext.str()+".png");
            Mat orginal=readImage("CompressedImages/reconstructedImages/output/"+convert.str()+ext.str()+".png");
            
            while(x<=10)
            {
                const char* window_name = "Person Detection Window";  // Create window
                namedWindow( window_name, WINDOW_AUTOSIZE );
                rectImg r=pq.top();
                pq.pop();
                cout<<r.decision<<endl;
                rectangle(orginal, r.r, Scalar(26,98,26));
                //            namedWindow( window_name, WINDOW_AUTOSIZE );
                //            imshow( window_name, orginal);
                //            waitKey(0); // Wait
                x++;
            }
            cc++;
            
            imwrite("CompressedImages/reconstructedImages/detected/"+convert.str()+ext.str()+".png", orginal);
            x=1;
        }
        c++;
        
    }
}
int main()
{
    
    //    Mat src= readImage("person_007.bmp");
    //    patchSelection(src);
    //    searchForPedestrian(); // IN BIG DATA BASE
    //    reNumberImages();
    //    printPositivePatches(src);
    //    NegativeImageCalc();
    //    trainAndSaveModel();
    //    detectInCompressions
    String inFilename= "1.jpg";
    String outFilename= "IMG_4728_1.JPG";
    predictWithModel(inFilename);
    Mat orginal=readImage(inFilename);
    int x=1;
    while(x<=10)
    {
        const char* window_name = "Person Detection Window";  // Create window
        namedWindow( window_name, WINDOW_AUTOSIZE );
        rectImg r=pq.top();
        pq.pop();
        cout<<r.decision<<endl;
        rectangle(orginal, r.r, Scalar(26,98,26));
                    namedWindow( window_name, WINDOW_AUTOSIZE );
                    imshow( window_name, orginal);
                    waitKey(0); // Wait
        // Blah
        x++;
    }
    imwrite(outFilename, orginal);
    cout<<"FINISHED"<<endl;
}

