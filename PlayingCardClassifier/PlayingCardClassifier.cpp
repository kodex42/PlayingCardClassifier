#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

int main() {
    int k = 10;
    int range = 45;
    double minThreshold = 75;
    double maxThreshold = 100;
    Mat img, filtered, edges;
    vector<vector<Point>> contours;

    // Prepare camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
    }

    while (true) {
        cap >> img;
        cvtColor(img, img, COLOR_BGR2GRAY);

        // Initialize
        filtered.create(img.size(), CV_8U);
        edges.create(img.size(), CV_8U);

        imshow("Source", img); // Display the source image

        // Apply smoothing
        bilateralFilter(img, filtered, k, range, range);
        //imshow("Bilateral Filtering", filtered); // Display the filtered image

        // Perform canny edge detection
        Canny(filtered, edges, minThreshold, maxThreshold);
        //imshow("Canny Edge Detector", edges); // Display the edge detection

        // Perform dilation to boost edges
        dilate(edges, edges, Mat(), Point(-1, -1), 1, 1, 1);
        //imshow("Dilated Edges", edges); // Display the edge detection

        // Detect rectangular contours
        findContours(edges, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        vector<RotatedRect> rects(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            rects[i] = minAreaRect(contours[i]);
        }
        // Draw rectangular contours
        cvtColor(img, img, COLOR_GRAY2BGR);
        for (size_t i = 0; i < contours.size(); i++) {
            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            Point2f rectPoints[4];
            rects[i].points(rectPoints);
            for (int j = 0; j < 4; j++) {
                line(img, rectPoints[j], rectPoints[(j + 1) % 4], color);
            }
        }
        imshow("Contours", img); // Display the contour detection
        
        // Wait 5 frames
        waitKey(5);
    }
    
    return 0;
}