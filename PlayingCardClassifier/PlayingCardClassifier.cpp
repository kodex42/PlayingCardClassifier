#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

const float EXPECTED_RATIO = 6.30f / 8.75f;
const float TOLERANCE = 0.15;
const float MINIMUM_SIZE = 5000;
const string KNOWN_CARDS[] = { "cards_0000_AH", "cards_0001_5H" , "cards_0002_QS" , "cards_0003_TH" , "cards_0004_AS" , "cards_0005_JK" , "cards_0006_QC" };
const string CARD_NAMES[] = { "Ace of Hearts", "Five of Hearts", "Queen of Spades", "Ten of Hearts", "Ace of Spades", "Joker", "Queen of Clubs" };
const int NUM_KNOWN_CARDS = 7;

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

int main() {
    int k = 8;
    int sigma = 55;
    double minThreshold = 75;
    double maxThreshold = 100;
    Mat img, filtered, edges, thresholded, withContours, out;
    vector<vector<Point>> contours;

    // Prepare camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
    }

    while (true) {
        cap >> img;
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
        threshold(filtered, thresholded, 200, 255, THRESH_BINARY);

        // Perform canny edge detection
        Canny(thresholded, edges, minThreshold, maxThreshold);
        //imshow("Canny Edge Detector", edges); // Display the edge detection

        // Perform dilation to boost edges
        dilate(edges, edges, Mat(), Point(-1, -1), 1, 1, 1);
        //imshow("Dilated Edges", edges); // Display the edge detection

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
        //imshow("Contours", withContours); // Display the contour detection

        try {
            // Extract contours
            vector<Mat> regions;
            vector<Point> middles;
            for (size_t i = 0; i < approximations.size(); i++) {
                // Store the first 3 points in a vector
                Point2f srcPoints[3];
                srcPoints[0] = approximations[i][0];
                srcPoints[1] = approximations[i][1];
                srcPoints[2] = approximations[i][2];

                // Warp the approxmated points to a 100x100 square image
                Point2f dstPoints[3];
                dstPoints[0] = Point(0, 0);
                dstPoints[1] = Point(0, 499);
                dstPoints[2] = Point(499, 499);


                Mat warpMatrix = getAffineTransform(srcPoints, dstPoints);
                Mat extraction;
                warpAffine(img, extraction, warpMatrix, Size(500, 500), INTER_LINEAR, BORDER_CONSTANT);

                // Store each image
                regions.push_back(extraction);
                middles.push_back(centroid(approximations[i]));
            }
        
            // Compare saved regions with known cards
            vector<double> cardDiffs;
            for (size_t i = 0; i < regions.size(); i++) {
                Mat region = regions[i];
                cardDiffs.clear();
                for (int k = 0; k < NUM_KNOWN_CARDS; k++) {
                    Mat card = imread("src/scans/" + KNOWN_CARDS[k] + ".jpg", IMREAD_GRAYSCALE);
                    vector<double> diffs;
                    for (int c = 0; c < 4; c++) {
                        // Try each orientation of all known cards and store their difference norms with the current region
                        rotate(card, card, ROTATE_90_CLOCKWISE);
                        //cout << "card size: " << card.size() << ", region size: " << region.size() << endl;
                        diffs.push_back(norm(region, card));
                    }
                    // Keep only the minimum difference
                    cardDiffs.push_back(*min_element(diffs.begin(), diffs.end()));
                }

                // Find the most likely card and draw its name to the appropriate card
                int minIndex = min_element(cardDiffs.begin(), cardDiffs.end()) - cardDiffs.begin();
                double min = cardDiffs[minIndex];
                Point loc = middles[i];
                string mostLikelyCard = CARD_NAMES[minIndex];
                int font = FONT_HERSHEY_COMPLEX;
                int thickness = 2;
                int baseline = 0;
                double fontScale = 0.75;
                Size textSize = getTextSize(mostLikelyCard, font, fontScale, thickness, &baseline);
                Point textOrg(loc.x - (textSize.width / 2), loc.y + (textSize.height / 2));
                putText(out, mostLikelyCard, textOrg, font, fontScale, Scalar(0, 0, 0), thickness + 1);
                putText(out, mostLikelyCard, textOrg, font, fontScale, Scalar(0, 255, 0), thickness);
            }

            imshow("Output", out); // Display the final detections
        }
        catch (cv::Exception& e) {
            cerr << e.what() << endl;
        }
        
        // Wait 5 frames
        waitKey(5);
    }
    
    return 0;
}