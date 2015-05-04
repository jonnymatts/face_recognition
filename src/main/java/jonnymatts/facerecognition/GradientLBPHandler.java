package jonnymatts.facerecognition;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class GradientLBPHandler {

	private int population;
	private LocalBinaryPatternHandler lbph;
	
	GradientLBPHandler(int p, int r) {
		population = p;
		lbph = new LocalBinaryPatternHandler(8, 1, true, false, false);
	}
	
	public static String getExtractorName(int population, int radius) {
		return "_GLBP_" + population + "_" + radius;
	}
	
	List<List<Double>> findFeatureVector(Mat colourImage, Mat depthImage) {
		
		// Find all the orientations needed for GLBP
		List<Double> orientationList = new ArrayList<Double>();
		double orientationDifference = 360 / population;
		for(double sum = 0; sum < 180; sum = (sum + orientationDifference)) {
			orientationList.add(sum);
		}
		
		// Find depth difference of image for each orientation
		List<Mat> depthDifferenceImageList = new ArrayList<Mat>();
		for(Double orientation : orientationList) {
			depthDifferenceImageList.add(findDepthDifferenceImageForOrientation(depthImage, orientation));
		}
		
		// Create histograms for depth difference images
		List<List<Double>> depthDifferenceHistogramList = new ArrayList<List<Double>>();
		for(Mat depthDifferenceImage : depthDifferenceImageList) {
			List<Double> imageHistogram = convertDepthDifferenceImageToHistogram(depthDifferenceImage);
			depthDifferenceHistogramList.add(imageHistogram);
		}
		
		List<List<Double>> lbpFeatureVector = lbph.findFeatureVector(colourImage, 8);
		
		depthDifferenceHistogramList.addAll(lbpFeatureVector);
		
		return depthDifferenceHistogramList;
	}
	
	private List<Double> convertDepthDifferenceImageToHistogram(Mat depthDifferenceImage) {
		List<Double> imageHistogram = new ArrayList<Double>();
		HashMap<Integer, Integer> differenceMap = new HashMap<Integer, Integer>();
		for(int i = 0; i < depthDifferenceImage.cols(); i++) {
			for(int j = 0; j < depthDifferenceImage.rows(); j++) {
				int val = (int)depthDifferenceImage.get(j, i)[0];
				if(differenceMap.containsKey(val)) {
					differenceMap.put(val, differenceMap.get(val) + 1);
				} else {
					differenceMap.put(val, 1);
				}
			}
		}
		for(int i = -8; i < 8; i++) {
			if(differenceMap.containsKey(i)) {
				imageHistogram.add((double)differenceMap.get(i));
			} else {
				imageHistogram.add(0d);
			}
		}
		return imageHistogram;
	}
	
	private Mat findDepthDifferenceImageForOrientation(Mat depthImage, double orientation) {
		Mat depthDifferenceImage = new Mat(depthImage.rows(), depthImage.cols(), CvType.CV_64FC1);
		
		// Get the index difference for the depth calculation
		int xDifference;
		int yDifference;
		if (orientation == 45.0) {
			xDifference = 1;
			yDifference = 1;
		} else if (orientation == 90.0) {
			xDifference = 1;
			yDifference = 0;
		} else if (orientation == 135.0) {
			xDifference = 1;
			yDifference = -1;
		} else {
			xDifference = 0;
			yDifference = 1;
		}
		
		for(int i = 0; i < depthImage.cols(); i++) {
			for(int j = 0; j < depthImage.rows(); j++) {
				int xIndexForDifference = i + xDifference;
				int yIndexForDifference = j + yDifference;
				if((xIndexForDifference > -1) && (xIndexForDifference < depthImage.cols()) && 
					(yIndexForDifference > -1) && (yIndexForDifference < depthImage.rows())) {
					double valToSubtract = depthImage.get(j, i)[0];
					double val = depthImage.get(yIndexForDifference, xIndexForDifference)[0];
					double imageValue = max(min(((int)val - valToSubtract), 7), -8);
					depthDifferenceImage.put(j, i, imageValue);
				} else {
					depthDifferenceImage.put(j, i, -9);
				}
			}
		}
		
		return depthDifferenceImage;
	}
}
