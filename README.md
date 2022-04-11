# Playing Card Classifier for COMP 4102

This program takes images of playing cards and attempts to classify them by using image processing and comparing processed images with scans of known cards.
This classifier can only recognize 7 different playing cards due to the difficulty in creatings high quality scans of playing cards.
Currently the program can only recognize the following cards:
1. Ace of Hearts
2. Five of Hearts
3. Queen of Spades
4. Ten of Hearts
5. Ace of Spades
6. Joker
7. Queen of Clubs

# Image Processing

Each image or webcam frame used as input goes through a 6 step image processing process:
1. Grayscale Conversion. Converting the input image to Grayscale helps with image processing.
2. Bilateral Filter. This helps to smooth the image for better processing while keeping edges sharp for detection. This is an alternative to Gaussian smoothing that is a better fit for finding edges later.
3. Threshold. Sets the value of pixels in the filtered image to 0 if their current value is not above a given threshold value. Playing cards are mostly white in color, so using a high threshold removes a lot of background noise that could compromise detection.
4. Canny Edge Detection. Finds edges in the thresholded image and sets pixels on edges as white and all other pixels black. This step creates a lot of false edges in the image, but should create a nice rectangular shape of edges around the playing card if the card in the image is on a contrasting background.
5. Edge Dilation. Enlarges edges. This step helps the following step to solidify the rectangular shape around the playing card as a full polygon with no breaks.
6. Contour Location. Finds all closed shapes among the dilated images. By locating all contours in the edges, the program can pick out which ones most look like a playing card.

# How the Program Knows what a Playing Card Looks Like

When the program computes a list of contours it converts them into rough quads of points representing a region in the image where a rectangle was detected.
Since images can have very complicated backgrounds, I decided to have the program throw away any contours that don't meet a minimum area requirement of 5000 pixels squared.
The size of each contour, however, is not a reliable metric for determining if a contour is card-shaped, so what determines if a contour is card-shaped?
Playing cards can come in all shapes and sizes, so this program is limited not only to the amount of playing cards I can accurately scan, but also the amount of different decks I own.
Many playing cards also have very different designs and patterns, so the playing cards I use for classification and testing have to be from the same deck.
The deck of cards I used for my program are almost exactly 6.3cm by 8.75cm, so by giving the program this information it can compare the ratio's of each detected contour and discard those that don't have a similar ratio within a 15% tolerance.
The result is the program being able to extract contours from the input image that pertain only to a playing card in the image.

# Playing Card Classification

This program has a 500x500 scan of each card it can identify.
When the program identifies a card-shaped object, it extracts the contents of the contour from the Grayscale of the original input image and warps it to a 500x500 image.
Since the detected card and the known card scans are of identical dimensions, they can be directly compared to find the closest match between the card and all known cards.
The detected playing card could be in an incorrect orientation of some multiple of 90 degrees, so each possible rotation of the detected playing card is compared with each known card.
When a detected card and a known card are compared, their L2 Norm is computed and stored.
The known card that produces the smallest L2 Norm when compared with the detected card is the class of the detected card, and so it is drawn on top of the detected contour the detected card was extracted from and shown to the user.

# Execution

To run this program, download and extract the latest release and run the executable in one of the following ways:

1. (Webcam Version).
This method is for testing with a webcam and your own set of cards.
Run the executable as is from the command line or by double clicking the executable.
The program will attempt to receive input from a connected webcam. If you do not have a webcam connected, this method of running the program will not work.
The program will display your webcam feed as well as some image processing steps made to each captured frame in real time.
If you hold a playing card up in front of the webcam, the program will attempt to identify and classify the card.
You can hold as many cards in front of the webcamera as you want.
To exit the program, press Ctrl+C in the terminal window that shows up on program start.

2. (Validation Dataset Version).
This method is for testing without a webcam.
Run the executable from the command line as './PlayingCardClassifier all'.
The program will run using each of the images located in src/input as input.
Each image used as input will display the results in various windows.
Pressing any key with any of the display images selected will move to the next input.
Once all input images are exhausted, the program exits.
