#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

// Card detection parameters
const float EXPECTED_RATIO = 6.30f / 8.75f;
const float TOLERANCE = 0.15;
const float MINIMUM_SIZE = 5000;

// Known card data
const string KNOWN_CARDS[] = { "cards_0000_AH", "cards_0001_5H" , "cards_0002_QS" , "cards_0003_TH" , "cards_0004_AS" , "cards_0005_JK" , "cards_0006_QC" };
const string CARD_NAMES[] = { "Ace of Hearts", "Five of Hearts", "Queen of Spades", "Ten of Hearts", "Ace of Spades", "Joker", "Queen of Clubs" };
const int NUM_KNOWN_CARDS = 7;

// For feature comparison
const float NN_MATCH_RATIO = 0.8f;

// Comparison methods
const int L2NORM_COMPARISON = 0;
const int FEATURE_COMPARISON = 1;
const int HISTOGRAM_COMPARISON = 2;

float widthOfContour(vector<Point> points) {
    return (norm(points[0] - points[1]) + norm(points[2] - points[3])) / 2;
}

float heightOfContour(vector<Point> points) {
    return (norm(points[0] - points[2]) + norm(points[1] - points[3])) / 2;
}

bool fitsRatio(vector<Point> points) {
    float x = widthOfContour(points);
    float y = heightOfContour(points);
    float detected = x / y;
    float error = abs(detected - EXPECTED_RATIO);
    return error < TOLERANCE;
}

bool fitsMinimumSize(vector<Point> points) {
    float x = widthOfContour(points);
    float y = heightOfContour(points);
    return x * y >= MINIMUM_SIZE;
}

bool isCardShaped(vector<Point> points) {
    return fitsRatio(points) && fitsMinimumSize(points);
}

Point centroid(vector<Point> points) {
    float x = 0;
    float y = 0;
    for (size_t i = 0; i < points.size(); i++) {
        x += points[i].x;
        y += points[i].y;
    }
    x /= points.size();
    y /= points.size();
    return Point2f(x, y);
}

// Fast and moderately accurate, but requires testing each possible orientation
double compareNorm(Mat img1, Mat img2) {
    double error = norm(img1, img2, NORM_L2);
    double similarity = error / ((double)img1.rows * (double)img1.cols);
    return similarity;
}

/*
// Orientation agnostic, but far too slow
double compareFeatures(InputArray img1, InputArray img2) {
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2, display;

    // Initialize the ORB detector
    Ptr<AKAZE> akaze = AKAZE::create();

    // Find the keypoints and descriptors
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    // Find matching keypoints
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<DMatch> matches;
    for (size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if (dist1 < NN_MATCH_RATIO * dist2) {
            matches.push_back(first);
        }
    }

    return -(double)matches.size();
}

// Fast and orientation agnostic, but unreliable.
double compareHistograms(Mat img1, Mat img2) {
    Mat hist1, hist2;
    int histSize[] = { 256 };
    int channels[] = { 0 };
    float ranges[] = { 0, 256 };
    const float* pranges[] = { ranges };

    calcHist(&img1, 1, channels, Mat(), hist1, 1, histSize, pranges);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX);
    calcHist(&img2, 1, channels, Mat(), hist2, 1, histSize, pranges);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX);

    return -compareHist(hist1, hist2, HISTCMP_CORREL);
}*/

void classifyAndDraw(InputOutputArray out, Mat* cards, vector<Mat> regions, vector<Point> middles) {
    // Compare saved regions with known cards
    vector<double> cardDiffs;
    for (size_t i = 0; i < regions.size(); i++) {
        Mat region = regions[i];
        cardDiffs.clear();
        for (int k = 0; k < NUM_KNOWN_CARDS; k++) {
            Mat card = cards[k];
            vector<double> diffs;
            for (int c = 0; c < 4; c++) {
                // Try each orientation of all known cards and store their difference norms with the current region
                rotate(card, card, ROTATE_90_CLOCKWISE);

                // Compare the images L2 norm to lock in the orientation
                diffs.push_back(compareNorm(card, region));
            }
            // Keep only the minimum difference
            cardDiffs.push_back(*min_element(diffs.begin(), diffs.end()));
            
        }

        // Find the most likely card and draw its name to the appropriate card
        int bestMatchIndex = min_element(cardDiffs.begin(), cardDiffs.end()) - cardDiffs.begin();
        Point loc = middles[i];
        string mostLikelyCard = CARD_NAMES[bestMatchIndex];
        int font = FONT_HERSHEY_COMPLEX;
        int thickness = 2;
        int baseline = 0;
        double fontScale = 0.75;
        Size textSize = getTextSize(mostLikelyCard, font, fontScale, thickness, &baseline);
        Point textOrg(loc.x - (textSize.width / 2), loc.y + (textSize.height / 2));
        putText(out, mostLikelyCard, textOrg, font, fontScale, Scalar(0, 0, 0), thickness + 1);
        putText(out, mostLikelyCard, textOrg, font, fontScale, Scalar(0, 255, 0), thickness);
    }
}

void processImage(Mat img, Mat *cards, int k, int sigma, int minThreshold, double minCanny, double maxCanny) {
    Mat filtered, edges, thresholded, withContours, out;
    vector<vector<Point>> contours;

    cvtColor(img, img, COLOR_BGR2GRAY);
    //cout << img.cols << " " << img.rows << endl;

    // Initialize
    filtered.create(img.size(), CV_8U);
    edges.create(img.size(), CV_8U);

    //imshow("Source", img); // Display the source image

    // Apply smoothing
    bilateralFilter(img, filtered, k, sigma, sigma);
    //imshow("Bilateral Filtering", filtered); // Display the filtered image

    // Apply threshold
    threshold(filtered, thresholded, minThreshold, 255, THRESH_BINARY);

    // Perform canny edge detection
    Canny(thresholded, edges, minCanny, maxCanny);
    //imshow("Canny Edge Detector", edges); // Display the edge detection

    // Perform dilation to boost edges
    dilate(edges, edges, Mat(), Point(-1, -1), 1, 1, 1);
    imshow("Dilated Edges", edges); // Display the edge detection

    // Detect contours and approximate as polygons
    vector<vector<Point>> approximations;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
        // Should have 4 vertices
        if (approx.size() == 4 && isCardShaped(approx))
            approximations.push_back(approx);
    }

    img.copyTo(withContours);
    cvtColor(img, withContours, COLOR_GRAY2BGR);
    drawContours(withContours, approximations, -1, Scalar(0, 0, 255), 2);
    withContours.copyTo(out);
    imshow("Contours", withContours); // Display the contour detection

    // Extract contours
    vector<Mat> regions;
    vector<Point> middles;
    for (size_t i = 0; i < approximations.size(); i++) {
        // Store each point in a vector
        Point2f srcPoints[4];
        srcPoints[0] = approximations[i][0];
        srcPoints[1] = approximations[i][1];
        srcPoints[2] = approximations[i][2];
        srcPoints[3] = approximations[i][3];

        // Warp the approxmated points to a 500x500 square image
        Point2f dstPoints[4];
        dstPoints[0] = Point(0, 0);
        dstPoints[1] = Point(0, 499);
        dstPoints[2] = Point(499, 499);
        dstPoints[3] = Point(499, 0);


        Mat warpMatrix = getPerspectiveTransform(srcPoints, dstPoints);
        Mat extraction;
        warpPerspective(img, extraction, warpMatrix, Size(500, 500), INTER_LINEAR, BORDER_CONSTANT);

        // Store each image
        regions.push_back(extraction);
        middles.push_back(centroid(approximations[i]));
    }

    for (size_t i = 0; i < regions.size(); i++) {
        imshow("Detected Region", regions[i]);
    }

    classifyAndDraw(out, cards, regions, middles);
    imshow("Output", out); // Display the final detections
}

void processArray(string* inputs, int n, Mat *loadedCards) {
    for (int i = 0; i < n; i++) {
        Mat img = imread("src/input/" + inputs[i] + ".jpg");

        processImage(img, loadedCards, 4, 15, 150, 75, 100);

        waitKey(0);
    }
}

int main(int argc, char **argv) {
    Mat img;
    Mat loadedCards[7];
    for (int i = 0; i < NUM_KNOWN_CARDS; i++)
        loadedCards[i] = imread("src/scans/" + KNOWN_CARDS[i] + ".jpg", IMREAD_GRAYSCALE);

    if (argc == 1) { // Normal mode: uses webcam
        // Prepare camera
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cout << "Cannot open camera" << endl;
        }

        while (true) {
            cap >> img;

            processImage(img, loadedCards, 8, 55, 200, 75, 100);

            // Wait 5 frames
            waitKey(5);
        }
    }
    else {
        string* testImages;
        int n;
        if (!string(argv[1]).compare("all")) {
            n = 11;
            testImages = new string[]{ "5H", "AH", "AH_5H_AS", "AS", "JK", "QC", "QS", "QS_JK", "TH", "TH_QC", "TH_QC_QS_JK" };
        }
        else {
            n = argc - 1;
            testImages = new string[n];
            for (int i = 0; i < n; i++)
                testImages[i] = string(argv[i + 1]);
        }
        processArray(testImages, n, loadedCards);
    }
    
    return 0;
}