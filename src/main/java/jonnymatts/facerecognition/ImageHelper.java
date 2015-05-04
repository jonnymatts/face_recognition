package jonnymatts.facerecognition;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static jonnymatts.facerecognition.ApplicationUtil.userDir;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class ImageHelper {
	
	public static List<LBPColour> lbpColourList = Arrays.asList(LBPColour.RED, LBPColour.GREEN, LBPColour.BLUE);
	
	public static Mat histogramEqualiseImage(Mat image) {
		List<Mat> channels = new ArrayList<Mat>();
		Mat equalizedImage = new Mat();
		
		// Convert image to HSV
		Imgproc.cvtColor(image, equalizedImage, Imgproc.COLOR_BGR2HSV);
		
		// Split image into three channels
		Core.split(equalizedImage,channels);
		
		// Eqaulise using V (intensity)
		Imgproc.equalizeHist(channels.get(2), channels.get(2));
		
		Core.merge(channels,equalizedImage);
		
		// Convert back to RGB
		Imgproc.cvtColor(equalizedImage, equalizedImage, Imgproc.COLOR_HSV2BGR);
		return equalizedImage;
	}
	
	public static List<Mat> preprocessImages(Mat colourImage, Mat depthImage, int finalImageDimension) {
		
		try {
			// Initialize classifier for faces
			CascadeClassifier cas = new CascadeClassifier("/Users/jonnymatts/dev/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml");
			
			// Convert colour image to greyscale
			Mat grey = new Mat();
			MatOfRect rectMat = new MatOfRect();
			Imgproc.cvtColor(colourImage, grey, Imgproc.COLOR_BGR2GRAY);

			// Detect face within image
			cas.detectMultiScale(grey, rectMat);
			
			// Create sub-images using face found from classifier
			if(rectMat.toList().isEmpty()) throw new CouldNotFindFaceException();
			Rect faceRect = rectMat.toList().get(0);
			colourImage = colourImage.submat(faceRect);
			depthImage = depthImage.submat(faceRect);
			
			// Histogram equalise colour image
			Mat equalisedColourImage = histogramEqualiseImage(colourImage);
			
			// Resize images to the needed dimension
			Imgproc.resize(equalisedColourImage, equalisedColourImage, new Size(finalImageDimension, finalImageDimension));
			Imgproc.resize(depthImage, depthImage, new Size(finalImageDimension, finalImageDimension));

			return Arrays.asList(equalisedColourImage, depthImage);
		} catch (Exception e) {
			return new ArrayList<Mat>();
		}
	}
	
	public static double featureMapSum(List<Mat> mapList, int i, int j) {
		double sum = 0;
		for(int k = 0; k < mapList.size(); k++) {
			sum += mapList.get(k).get(j, i)[0];
		}
		return sum;
	}
	
	public static List<Mat> resizeFeatureMaps(List<Mat> mapList, int dimension) {
		return mapList.stream().map(m -> resizeImageIfNeeded(m, dimension)).collect(Collectors.toList());
	}
	
	public static Mat resizeImageIfNeeded(Mat img, int dimension) {
		Mat returnImage = img;
		if(img.rows() != dimension) Imgproc.resize(img, returnImage, new Size(dimension, dimension));
		return returnImage;
	}
	
	public static List<Mat> normaliseFeatureMapList(List<Mat> mapList, int localNeighbourhoodSize, double maxima, double differenceCutoff) {
		return mapList.stream().map(m -> normaliseFeatureMap(m, localNeighbourhoodSize, maxima, differenceCutoff)).collect(Collectors.toList());
	}
	
	public static Mat normaliseFeatureMap(Mat img, int localNeighbourhoodSize, double maxima, double differenceCutoff) {
		
		// Find smallest and highest value within image
		double smallestValue = findSmallestValueForSingleChannelImage(img);
		double highestValue = findHighestValueForSingleChannelImage(img);
		
		// Normalise values between 0 and maxima, and find index of global maxima
		Mat normalisedImage = img.clone();
		int globalMaximaX = -1;
		int globalMaximaY = -1;
		double multiplicationFactor = maxima / highestValue;
		double newHighestValue = 0;
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				double val = ((img.get(j, i)[0] - smallestValue) * multiplicationFactor);
				if(val > newHighestValue) {
					globalMaximaX = i;
					globalMaximaY = j;
					newHighestValue = val;
				}
				normalisedImage.put(j, i, val);
			}
		}
		
		// Needed to ensure that resolution is not lost in return image, no idea why
		Mat2BufferedImage(normalisedImage);
		
		// Find mean of local maxima values around global maxima
		int nBoundary = ((localNeighbourhoodSize + 1) / 2) - 1; 
		int nXMinIndex = max(0, (globalMaximaX - nBoundary));
		int nXMaxIndex = min(img.cols(), (globalMaximaX + nBoundary));
		int nYMinIndex = max(0, (globalMaximaY - nBoundary));
		int nYMaxIndex = min(img.rows(), (globalMaximaY + nBoundary));
		
		double localSum = 0;
		int localCount = 0;
		for(int k = nXMinIndex; k < nXMaxIndex; k++) {
			for(int l = nYMinIndex; l < nYMaxIndex; l++) {
				double val = normalisedImage.get(l, k)[0];
				localSum += val;
				localCount++;
			}
		}
		double localAverage = localSum / localCount;
		
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				// Get pixel value
				double pixelVal = normalisedImage.get(j, i)[0];
				double localDifference = abs(pixelVal - localAverage);
				double newVal = (localDifference > differenceCutoff) ? (pixelVal) : 0;
				normalisedImage.put(j, i, newVal);
			}
		}
		
		return normalisedImage;
	}
	
	public static Mat normaliseDepthImage(Mat img) {
		Mat returnImg = new Mat(img.rows(), img.cols(), CvType.CV_64FC1);
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				double val = img.get(j, i)[0];
				if(val == 255d) {
					returnImg.put(j, i, 254.0);
				} else {
					returnImg.put(j, i, val);
				}
			}
		}
		return returnImg;
	}
	
	public static List<Mat> findGaborPyramidForImage(Mat img, int scale, int kSize, double sigma, double theta,
													 double lambda, double gamma) {
		List<Mat> gaborPyramid = new ArrayList<Mat>();
		gaborPyramid.add(img);
		Mat currentImage = img;
		for(int i = 0; i < scale; i++) {
			Mat filteredImage = new Mat();
			Mat reducedImage = new Mat();
			Mat gKernel = Imgproc.getGaborKernel(new Size(kSize, kSize), sigma, theta, lambda, gamma);
			Imgproc.filter2D(currentImage, filteredImage, CvType.CV_64F, gKernel);
			Imgproc.resize(filteredImage, reducedImage, new Size(currentImage.cols()/2, currentImage.rows()/2));
			gaborPyramid.add(reducedImage);
			currentImage = reducedImage;
		}
		return gaborPyramid;
	}
	
	public static List<Mat> findGaussianPyramidForImage(Mat img, int scale) {
		List<Mat> gaussianPyramid = new ArrayList<Mat>();
		gaussianPyramid.add(img);
		Mat currentImage = img;
		for (int i = 0; i < scale; i++) {
			Mat reducedImage = new Mat();
			Imgproc.pyrDown(currentImage, reducedImage);
			gaussianPyramid.add(reducedImage);
			currentImage = reducedImage;
		}
		return gaussianPyramid;
	}
	
	public static Mat normaliseEntropyImage(Mat img) {
		
		Mat returnImage = img.clone();
		
		// Find highest value in list
		List<Double> valueList = convertMatToList(returnImage, 0);
		List<Double> sortedList = valueList.stream().sorted((a, b) -> Double.compare(a, b)).collect(Collectors.toList());
		
		// Shift all values so smallest value is always 0, multiply so max is 255
		Double smallestValue = sortedList.get(0);
		Double highestValue = sortedList.get(sortedList.size() - 1);
		Double multiplicationFactor = 255 / highestValue;
		
		if(smallestValue != 0) {
			for(int i = 0; i < img.cols(); i++) {
				for(int j = 0; j < img.rows(); j++) {
					if(smallestValue < 0) {
						returnImage.put(j, i, ((returnImage.get(j, i)[0] - smallestValue) * multiplicationFactor));
					} else {
						returnImage.put(j, i, ((returnImage.get(j, i)[0] + smallestValue) * multiplicationFactor));
					}
				}
			}
		}
		
		return returnImage;
	}
	
	public static Mat convertImageToSingleRowMatrix(Mat img, int channel) {
		
		// Not sure why, but needed otherwise returnImage is emtpy???
		img.dump();
		
		Mat returnImage = new Mat(1, (int)img.size().area(), CvType.CV_64FC1);
		
		int index = 0;
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				returnImage.put(1, index, img.get(j, i)[channel]);
			}
		}
		
		return returnImage;
	}
	
	public static List<Mat> convertMultiChannelImageToImageList(Mat img) {
		Mat blueImg = new Mat(img.rows(), img.cols(), CvType.CV_64FC1);
		Mat greenImg = new Mat(img.rows(), img.cols(), CvType.CV_64FC1);
		Mat redImg = new Mat(img.rows(), img.cols(), CvType.CV_64FC1);
		
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				blueImg.put(j, i, img.get(j, i)[0]);
				greenImg.put(j, i, img.get(j, i)[1]);
				redImg.put(j, i, img.get(j, i)[2]);
			}
		}
		
		return Arrays.asList(blueImg, greenImg, redImg);
	}
	
	public static Mat subtractSingleChannelImages(Mat img1, Mat img2) {
		Mat subtractedImage = img1.clone();
		for(int i = 0; i < img1.cols(); i++) {
			for(int j = 0; j < img1.rows(); j++) {
				subtractedImage.put(j, i, (img1.get(j, i)[0] - img2.get(j, i)[0]));
			}
		}
		return subtractedImage;
	}
	
	public static Mat subtractImageFromMultiChannelImage(Mat img, Mat channel1, Mat channel2, Mat channel3) {
		Mat subtractedImage = img.clone();
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				double[] vals = subtractedImage.get(j, i);
				double[] newVals = {(vals[0] - channel1.get(j, i)[0]), (vals[1] - channel2.get(j, i)[0]),
									(vals[2] - channel3.get(j, i)[0])};
				subtractedImage.put(j, i, newVals);
			}
		}
		return subtractedImage;
	}
	
	public static Mat convertImageListToSingleMatrix(List<Mat> imgList, int rows, int cols) {
		Mat returnMatrix = new Mat(rows, cols, CvType.CV_64F);
		int rowIndex = 0;
		// Create matrix containing all images, one per row
		for(Mat im : imgList) {
			List<Double> imValList = convertMatToList(im, 0);
			for(int i = 0; i < imValList.size(); i++) {
				returnMatrix.put(rowIndex, i, imValList.get(i));
			}
			rowIndex++;
		}
		return returnMatrix;
	}
	
	public static double findSmallestValueForSingleChannelImage(Mat img) {
		double smallestValue = 255;
		for (int i = 0; i < img.cols(); i++) {
			for (int j = 0; j < img.rows(); j++) {
				double val = img.get(j, i)[0];
				if(val < smallestValue) smallestValue = val;
			}
		}
		return smallestValue;
	}
	
	public static double findHighestValueForSingleChannelImage(Mat img) {
		double highestValue = 0;
		for (int i = 0; i < img.cols(); i++) {
			for (int j = 0; j < img.rows(); j++) {
				double val = img.get(j, i)[0];
				if(val > highestValue) highestValue = val;
			}
		}
		return highestValue;
	}
	
	public static boolean compareImages(Mat im1, Mat im2) {
		Mat op = new Mat();
		Core.compare(im1, im2, op, Core.CMP_NE);
		return (Core.countNonZero(op) == 0);
	}
	
	public static Mat convertRGBtoYUV(Mat in) {
		Mat yuvImage = new Mat(in.rows(), in.cols(), CvType.CV_32FC3);
		
		for (int i = 0; i < in.cols(); i++) {
			for (int j = 0; j < in.rows(); j++) {
				double[] bgrArray = in.get(j,i);
				double bVal = bgrArray[0];
				double gVal = bgrArray[1];
				double rVal = bgrArray[2];
				float yVal = (float)((0.29f * rVal) + (0.587f * gVal) + (0.114f * bVal));
				float uVal = (float)(0.492f * (bVal - yVal));
				float vVal = (float)(0.877f * (rVal - yVal));
				float[] yuvArray = {yVal, uVal, vVal};
				yuvImage.put(j, i, yuvArray);
			}
		}
		
		return yuvImage;
	}

	public static double bilinearInterpolation(double x, double y, double xMin, double xMax, double yMin, double yMax, double x1, double x2, double x3, double x4) {

		double xDistToMax = (xMax - x) / (xMax - xMin);
		double xDistToMin = (x - xMin) / (xMax - xMin);
		double yDistToMax = (yMax - y) / (yMax - yMin);
		double yDistToMin = (y - yMin) / (yMax - yMin);

		double returnVal;

		// Y-axis interpolation
		if (x % 1 == 0) {
			returnVal = (yDistToMax * x1) + (yDistToMin * x3);
		}
		// X-axis interpolation
		else if (y % 1 == 0) {
			returnVal = (xDistToMax * x1) + (xDistToMin * x2);
		}
		// X and Y-axis interpolation
		else {

			// Find two interpolated points down the x-axis
			double xInt1 = (xDistToMax * x1) + (xDistToMin * x2);
			double xInt2 = (xDistToMax * x3) + (xDistToMin * x4);

			// Interpolate between the two points
			double yInt = (yDistToMax * xInt1) + (yDistToMin * xInt2);

			returnVal = yInt;
		}
		return returnVal;
	}

	// Converts 2D matrix to 1D list
	public static List<Double> convertMatToList(Mat input, int index) {
		List<Double> returnList = new ArrayList<Double>();

		for (int i = 0; i < input.rows(); i++) {
			for (int j = 0; j < input.cols(); j++) {
				returnList.add(new Double(input.get(i, j)[index]));
			}
		}
		return returnList;
	}

	public static Mat readImageFromFile(String filename) {
		Mat img = Highgui.imread(userDir + filename);
		return img;
	}
	
	public static File loadFile(String filename) {
		return new File(userDir + filename);
	}

	// Applies the supplied feature detector to the suppplied image
	public static Mat useFeatureDetector(Mat img, CascadeClassifier cas) {
		Mat grey = new Mat();
		MatOfRect rectMat = new MatOfRect();

		// Convert colour image to greyscale
		Imgproc.cvtColor(img, grey, Imgproc.COLOR_BGR2GRAY);

		// Use detector on greyscale image
		cas.detectMultiScale(grey, rectMat);

		// For each face detected, draw rectangle onto original image
		for (Rect rect : rectMat.toList()) {
			Core.rectangle(img, new Point(rect.x, rect.y), new Point(rect.x
					+ rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
		}

		return img;
	}
	
	public static void displayImageList(List<Mat> imgList) {
		for(Mat img : imgList) {
			displayImage(img);
		}
	}

	public static void displayImage(Mat mat) {
		Image img = Mat2BufferedImage(mat);
		ImageIcon icon = new ImageIcon(img);
		JFrame frame = new JFrame();
		frame.setLayout(new FlowLayout());
		frame.setSize(img.getWidth(null) + 50, img.getHeight(null) + 50);
		JLabel lbl = new JLabel();
		lbl.setIcon(icon);
		frame.add(lbl);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static BufferedImage Mat2BufferedImage(Mat m) {
		m.convertTo(m, CvType.CV_8U);
		int type = BufferedImage.TYPE_BYTE_GRAY;
		if (m.channels() > 1) {
			type = BufferedImage.TYPE_3BYTE_BGR;
		}
		int bufferSize = m.channels() * m.cols() * m.rows();
		byte[] b = new byte[bufferSize];
		m.get(0, 0, b); // get all the pixels
		BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster()
				.getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);
		return image;
	}
}
