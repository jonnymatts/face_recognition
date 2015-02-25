package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.*;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class LocalBinaryPatternHandler {
	
	// Find the feature vector for given image, population and radius
	public List<List<Double>> findFeatureVector(Mat img, int windowSize, int population, int radius) {
		
		// Split image into 5x5 grid of sub-images
		List<Mat> subImgList = new ArrayList<Mat>();
		int subImgWidth = (int)Math.floor(img.cols()/5);
		int subImgHeight = (int)Math.floor(img.rows()/5);
		
		for(int i = 0; i < 5; i++) {
			for(int j = 0; j < 5; j++) {
				int iIndex = i * subImgWidth;
				int jIndex = j * subImgHeight;
				subImgList.add(img.submat(iIndex, iIndex+subImgWidth, jIndex, jIndex+subImgHeight));
			}
		}
		
		// Find histogram for each sub-image
		List<List<Double>> histList = new ArrayList<List<Double>>();
		int a = 0;
		int i = 0;
		for(Mat subImg : subImgList) {
			
			// Create histogram list for sub-image
			int histSize = (int)Math.pow(2, (double)population);
			Double[] histArr = new Double[histSize];
			
			// Fill histogram with default values (0)
			Arrays.fill(histArr, 0d);
			List<Double> imgHist = Arrays.asList(histArr);
			Collections.fill(imgHist, 0d);
			
			// Find the LBP value at each pixel
			Mat lbpImg = calculateLBP(subImg, windowSize, population, radius);
			
			List<Double> lbpValueList = convertMatToList(lbpImg);
			
			// Limit values between 0 and maxValue 
			lbpValueList = lbpValueList.stream().map(v -> Math.max(0, Math.min(histSize-1, v))).collect(Collectors.toList());
			
			// Populate histogram with LBP data
			for(Double val : lbpValueList) {
				Integer iVal = val.intValue();
				imgHist.set(iVal, (imgHist.get(iVal) + 1));
			}
			
			histList.add(imgHist);
			i++;
		}
		
		return histList;
	}
	
	// Calculate the LBP for each pixel within the image
	Mat calculateLBP(Mat img, int windowSize, int population, int radius) {
		Mat lbpImg = new Mat(img.rows(), img.cols(), CvType.CV_64F);;
		int windowBoundary = ((windowSize + 1) / 2) - 1;
		
		for(int i = windowBoundary; i < (img.rows() - windowBoundary); i++) {
			for(int j = windowBoundary; j < (img.cols() - windowBoundary); j++) {
				if((i < windowBoundary) || (i > img.rows() - windowBoundary) 
						|| (j < windowBoundary) || (j > img.cols() - windowBoundary)) {
					lbpImg.put(i, j, 0);
				}
				lbpImg.put(i, j, calculateLBPForPixel(img, i, j, windowSize));
			}
		}
		
		return lbpImg;
	}
	
	Double calculateLBPForPixel(Mat img, int x, int y, int windowSize) {
		
		// Find radius of neighbourhood
		int nbhRadius = ((windowSize + 1) / 2) - 1;
		
		// Find neighbourhood for central pixel
		Mat nbh = img.submat(x-nbhRadius, x+nbhRadius+1, y-nbhRadius, y+nbhRadius+1);
		
		// Convert nbh to 1D list
		List<Double> nbhList = convertMatToList(nbh);
		
		// Remove and preserve central pixel value from list
		int centrePixelIndex = (windowSize * windowSize) / 2;
		Double centrePixelValue = nbhList.get(centrePixelIndex);
		nbhList.remove(centrePixelIndex);
		
		// Convert list value to either 1 or 0
		// if(value < centrePixelValue) 0 else 1
		List<Integer> nbhBinList = nbhList.stream().map(d -> (d < centrePixelValue)).map(b -> b ? 0 : 1).collect(Collectors.toList());;
		
		// Find the sum of values multiplied by the corresponding power of 2
		Double sum = 0d;
		for(int i = 0; i < nbhList.size(); i++){
			sum = sum + nbhBinList.get(i) * (Math.pow(2, i));
		}
		
		return sum;
	}

}
