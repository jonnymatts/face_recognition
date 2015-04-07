package jonnymatts.facerecognition;

import static java.lang.Math.*;
import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class RISEHandler {

	private int radius;
	private int kSize;
	private double sigma;
	private double lambda;
	private double gamma;
	private int localNeighbourhoodSize;
	private double maxima;
	private double differenceCutoff;
	
	public RISEHandler(int r, int ks, double s, double l, double g, int lns, double m, double lmc) {
		radius = r;
		kSize = ks;
		sigma = s;
		lambda = l;
		gamma = g;
		localNeighbourhoodSize = lns;
		maxima = m;
		differenceCutoff = lmc;
	}
	
	public List<Double> findFeatureVector(Mat image, Mat depthImage) {
		List<Double> returnList = new ArrayList<Double>();
		
		// Find the four entropy maps of the image
		List<Mat> entropyMapList = new ArrayList<Mat>();
		int width = ((image.cols() * 3) / 4);
		int height = ((image.rows() * 3) / 4);
		entropyMapList.add(findEntropyImageForImageSubsection(image, (image.cols() / 2), (image.rows() / 2)));
		entropyMapList.add(findEntropyImageForImageSubsection(image, height, width));
		entropyMapList.add(findEntropyImageForImageSubsection(depthImage, (image.cols() / 2), (image.rows() / 2)));
		entropyMapList.add(findEntropyImageForImageSubsection(depthImage, height, width));
		
		// Find the saliency map of the image
		Mat saliencyMap = findSaliencyMap(image);
		
		displayImageList(entropyMapList);
		displayImage(saliencyMap);
		
		return returnList;
	}
	
	private Mat createColourFeatureMap(List<Mat> pyramid1, List<Mat> pyramid2, int c, int d) {
		Mat image = subtractSingleChannelImages(pyramid1.get(c), pyramid2.get(c));
		Mat imageToSubtract = subtractSingleChannelImages(pyramid2.get((c + d)), pyramid1.get((c + d)));
		return acrossScaleDifference(image, imageToSubtract, d);
	}
	
	private Mat createFeatureMap(List<Mat> pyramid, int c, int d) {
		Mat image = pyramid.get(c);
		Mat imageToSubtract = pyramid.get((c + d));
		return acrossScaleDifference(image, imageToSubtract, d);
	}
	
	private Mat acrossScaleDifference(Mat image, Mat imageToSubtract, int d) {
		Mat returnImage = new Mat(image.rows(), image.cols(), CvType.CV_64F);

		Mat tempMat = new Mat();
		// Ensure images are both the same size
		for (int i = 0; i < d; i++) {
			Imgproc.pyrUp(imageToSubtract, tempMat);
			imageToSubtract = tempMat;
		}

		for (int i = 0; i < image.cols(); i++) {
			for (int j = 0; j < image.rows(); j++) {
				double val = abs(image.get(j, i)[0]
						- imageToSubtract.get(j, i)[0]);
				returnImage.put(j, i, val);
			}
		}

		return returnImage;
	}
	
	Mat findSaliencyMap(Mat img) {
		
		Mat returnImage = new Mat();
		
		// Find intensity, Red, Green, Blue and Yellow images
		Mat intensityImage = new Mat(img.rows(), img.cols(), CvType.CV_64F);
		Mat redImage = new Mat(img.rows(), img.cols(), CvType.CV_64F);
		Mat greenImage = new Mat(img.rows(), img.cols(), CvType.CV_64F);
		Mat blueImage = new Mat(img.rows(), img.cols(), CvType.CV_64F);
		Mat yellowImage = new Mat(img.rows(), img.cols(), CvType.CV_64F);
		for(int i = 0; i < img.cols(); i++) {
			for(int j = 0; j < img.rows(); j++) {
				double[] iValues = img.get(j, i);
				double b = iValues[0]; double g = iValues[1]; double r = iValues[2];
				intensityImage.put(j, i, ((b + g + r) / 3));
				redImage.put(j, i, (r - ((g + b) / 2)));
				greenImage.put(j, i, (g - ((r + b) / 2)));
				blueImage.put(j, i, (b - ((g + r) / 2)));
				yellowImage.put(j, i, max((((r + g) / 2) - abs((r - g) / 2)), 0));
			}
		}
		
		// Find Gaussian pyramids
		List<Mat> intensityGaussianPyramid = findGaussianPyrmaidForImage(intensityImage, 8);
		List<Mat> redGaussianPyramid = findGaussianPyrmaidForImage(redImage, 8);
		List<Mat> greenGaussianPyramid = findGaussianPyrmaidForImage(greenImage, 8);
		List<Mat> blueGaussianPyramid = findGaussianPyrmaidForImage(blueImage, 8);
		List<Mat> yellowGaussianPyramid = findGaussianPyrmaidForImage(yellowImage, 8);
		
		// Extract intensity and colour feature maps
		List<Mat> intensityFeatureMaps = new ArrayList<Mat>();
		List<Mat> rgColourFeatureMaps = new ArrayList<Mat>();
		List<Mat> byColourFeatureMaps = new ArrayList<Mat>();
		for(int c = 2; c < 5; c++) {
			for(int d = 3; d < 5; d++) {
				intensityFeatureMaps.add(createFeatureMap(intensityGaussianPyramid, c, d));
				rgColourFeatureMaps.add(createColourFeatureMap(redGaussianPyramid, greenGaussianPyramid, c, d));
				byColourFeatureMaps.add(createColourFeatureMap(blueGaussianPyramid, yellowGaussianPyramid, c, d));
			}
		}
		
		// Find orientation gabor pyramids
		List<List<Mat>> orientationGaborPyramids = new ArrayList<List<Mat>>();
		double[] thetaArray = {0, 45, 90, 135};
		for(int i = 0; i < thetaArray.length; i++) {
			orientationGaborPyramids.add(findGaborPyramidForImage(img, 8, kSize, sigma, thetaArray[i], lambda, gamma));
		}
		
		// Extract orientation feature maps
		List<List<Mat>> orientationFeatureMaps = new ArrayList<List<Mat>>();
		for(int i = 0; i < thetaArray.length; i++){
			List<Mat> orientationFeatureMap = new ArrayList<Mat>(); 
			for(int c = 2; c < 5; c++) {
				for(int d = 3; d < 5; d++) {
					orientationFeatureMap.add(createFeatureMap(orientationGaborPyramids.get(i), c, d));
				}
			}
			orientationFeatureMaps.add(orientationFeatureMap);
		}
		
		// Normalise all the feature maps
		List<Mat> normalisedIntensityFeatureMaps = normaliseFeatureMapList(intensityFeatureMaps, localNeighbourhoodSize, maxima, differenceCutoff);
		List<Mat> normalisedRGColourFeatureMaps = normaliseFeatureMapList(rgColourFeatureMaps, localNeighbourhoodSize, maxima, differenceCutoff);
		List<Mat> normalisedBYColourFeatureMaps = normaliseFeatureMapList(byColourFeatureMaps, localNeighbourhoodSize, maxima, differenceCutoff);
		List<List<Mat>> normalisedOrientationFeatureMaps = orientationFeatureMaps.stream().map(l -> normaliseFeatureMapList(l, localNeighbourhoodSize, maxima, differenceCutoff)).collect(Collectors.toList());
		
		// Find intensity conspicuity map
		int dimension = intensityGaussianPyramid.get(4).cols();
		Mat intensityConspicuityMap = new Mat(dimension, dimension, CvType.CV_64F);
		List<Mat> resizedNormalisedIntensityFeatureMaps = resizeFeatureMaps(normalisedIntensityFeatureMaps, dimension);
		for (int i = 0; i < intensityConspicuityMap.cols(); i++) {
			for (int j = 0; j < intensityConspicuityMap.rows(); j++) {
				intensityConspicuityMap.put(j, i, featureMapSum(resizedNormalisedIntensityFeatureMaps, i, j));
			}
		}
		
		// Find colour conspicuity map
		Mat colourConspicuityMap = new Mat(dimension, dimension, CvType.CV_64F);
		List<Mat> resizedNormalisedRGColourFeatureMaps = resizeFeatureMaps(normalisedRGColourFeatureMaps, dimension);
		List<Mat> resizedNormalisedBYColourFeatureMaps = resizeFeatureMaps(normalisedBYColourFeatureMaps, dimension);
		for (int i = 0; i < colourConspicuityMap.cols(); i++) {
			for (int j = 0; j < colourConspicuityMap.rows(); j++) {
				colourConspicuityMap.put(j, i, (featureMapSum(resizedNormalisedRGColourFeatureMaps, i, j)) + featureMapSum(resizedNormalisedBYColourFeatureMaps, i, j));
			}
		}
		
		// Find intermediary conspicuity maps for each orientation
		List<Mat> intermediaryOrientationConspicuityMaps = new ArrayList<Mat>();
		for(List<Mat> mapList : normalisedOrientationFeatureMaps) {
			Mat intermediaryOrientationConspicuityMap = new Mat(dimension, dimension, CvType.CV_64F);
			List<Mat> resizedOrientationFeatureMaps = resizeFeatureMaps(mapList, dimension);
			for (int i = 0; i < intermediaryOrientationConspicuityMap.cols(); i++) {
				for (int j = 0; j < intermediaryOrientationConspicuityMap.rows(); j++) {
					intermediaryOrientationConspicuityMap.put(j, i, featureMapSum(resizedOrientationFeatureMaps, i, j));
				}
			}
			intermediaryOrientationConspicuityMaps.add(intermediaryOrientationConspicuityMap);
		}
		
		// Find orientation conspicuity map
		List<Mat> normalisedIntermediaryOrientationConspicuityMaps = normaliseFeatureMapList(intermediaryOrientationConspicuityMaps, localNeighbourhoodSize, maxima, differenceCutoff);
		Mat orientationConspicuityMap = new Mat(dimension, dimension, CvType.CV_64F);
		for (int i = 0; i < orientationConspicuityMap.cols(); i++) {
			for (int j = 0; j < orientationConspicuityMap.rows(); j++) {
				double val = featureMapSum(normalisedIntermediaryOrientationConspicuityMaps, i, j);
				orientationConspicuityMap.put(j, i, val);
			}
		}
		
		// Create final saliency image
		Mat saliencyImage = new Mat(dimension, dimension, CvType.CV_64F);
		for (int i = 0; i < saliencyImage.cols(); i++) {
			for (int j = 0; j < saliencyImage.rows(); j++) {
				double intensityVal = normaliseFeatureMap(intensityConspicuityMap, localNeighbourhoodSize, maxima, differenceCutoff).get(j, i)[0];
				double colourVal = normaliseFeatureMap(colourConspicuityMap, localNeighbourhoodSize, maxima, differenceCutoff).get(j, i)[0];
				double orientationVal = normaliseFeatureMap(orientationConspicuityMap, localNeighbourhoodSize, maxima, differenceCutoff).get(j, i)[0];
				double val = ((intensityVal / 3) + (colourVal / 3) + (orientationVal / 3));
				saliencyImage.put(j, i, val);
			}
		}
		
		Imgproc.resize(saliencyImage, returnImage, new Size(img.rows(), img.cols()));
		
		return returnImage;
	}
	
	Mat findEntropyImageForImageSubsection(Mat img, int subsectionHeight, int subsectionWidth) {
		
		// Convert image to greyscale
		img.convertTo(img, CvType.CV_8U);
		
		Mat entropyImage = new Mat(subsectionHeight, subsectionWidth, CvType.CV_64F);
		
		// find centre pixel of image
		int xCentre = (int)floor(img.cols() / 2);
		int yCentre = (int)floor(img.rows() / 2);
		
		int xBoundary = ((subsectionWidth + 1) / 2) - 1;
		int yBoundary = ((subsectionHeight + 1) / 2) - 1;

		int entropyIIndex = 0;
		int entropyJIndex = 0;
		for (int i = xCentre - xBoundary; i < xCentre + xBoundary + 2; i++) {
			for (int j = yCentre - yBoundary; j < yCentre + yBoundary + 2; j++) {
				entropyImage.put(entropyIIndex, entropyJIndex, calculateEntropyForPixel(img, i, j));
				entropyJIndex++;
			}
			entropyJIndex = 0;
			entropyIIndex++;
		}
		
		return normaliseEntropyImage(entropyImage);
	}
	
	double calculateEntropyForPixel(Mat img, int x, int y) {
		
		int nBoundary = ((radius + 1) / 2) - 1;
		
		// Find histogram values for neighbourhood
		int[] nHist = new int[255];
		for(int i = y - nBoundary; i < y + nBoundary + 2; i++) {
			for(int j = x - nBoundary; j < x + nBoundary + 2; j++) {
				int intensityValue = (int)img.get(j, i)[0];
				nHist[intensityValue] = nHist[intensityValue] + 1;
			}
		}
		
		// Calculate entropy
		double entropy = 0;
		for(int i = 0; i < 255; i++) {
			double pmf = nHist[i] / (pow(radius, 2));
			double logValue = (log(pmf) / log(2));
			double entropyValue = (pmf == 0.0) ? 0 : (pmf * logValue);
			entropy += entropyValue;
		}
		
		entropy *= -1;
		
		return entropy;
	}
	
}
