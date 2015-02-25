package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;
import static java.lang.Math.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.*;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class LocalBinaryPatternHandler {
	
	// Find the feature vector for given image, population and radius
	public List<List<Double>> findFeatureVector(Mat img, int population, int radius) {
		
		// Split image into 5x5 grid of sub-images
		List<Mat> subImgList = new ArrayList<Mat>();
		int subImgWidth = (int)floor(img.cols()/5);
		int subImgHeight = (int)floor(img.rows()/5);
		
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
			int histSize = (int)pow(2, (double)population);
			Double[] histArr = new Double[histSize];
			
			// Fill histogram with default values (0)
			Arrays.fill(histArr, 0d);
			List<Double> imgHist = Arrays.asList(histArr);
			Collections.fill(imgHist, 0d);
			
			// Find the LBP value at each pixel
			Mat lbpImg = calculateLBP(subImg, population, radius);
			
			List<Double> lbpValueList = convertMatToList(lbpImg);
			
			// Limit values between 0 and maxValue 
			lbpValueList = lbpValueList.stream().map(v -> max(0, min(histSize-1, v))).collect(Collectors.toList());
			
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
	Mat calculateLBP(Mat img, int population, int radius) {
		Mat lbpImg = new Mat(img.rows(), img.cols(), CvType.CV_64F);
		
		for(int i = radius; i < (img.cols() - radius); i++) {
			for(int j = radius; j < (img.rows() - radius); j++) {
				if((i < radius) || (i > img.cols() - radius) 
						|| (j < radius) || (j > img.rows() - radius)) {
					lbpImg.put(j, i, 0);
				}
				lbpImg.put(j, i, calculateLBPForPixel(img, i, j, population, radius));
			}
		}
		
		return lbpImg;
	}
	
	// Calculate the LBP for each pixel within the image
	Double calculateLBPForPixel(Mat img, int x, int y, int population, int radius) {
		
		// Find neighbourhood for central pixel
		Mat nbh = img.submat(y-radius, min((y+radius+1), img.cols()), x-radius, min((x+radius+1), img.cols()));
		
		List<Double> nbhList = new ArrayList<Double>();
		
		// Use linear interpolation to find circular LBP values
		double anglePortion = (2 * PI) / population;
		for(int i = 0; i < population; i++) {
			double angle = i*anglePortion;
			double px = x + (radius * cos(angle));
			double py = y + (radius * sin(angle));
			double value = ((px == 0) || (py == 0) || ((px % 1 == 0) && (py % 1 ==0))) ? img.get((int)round(py), (int)round(px))[0] : bilinearInterpolation(img, px, py);
			nbhList.add(value);
		}
		
		// Remove and preserve central pixel value from list
		Double centrePixelValue = img.get(y, x)[0];
		
		// Convert list value to either 1 or 0
		// if(value < centrePixelValue) 0 else 1
		List<Integer> nbhBinList = nbhList.stream().map(d -> (d < centrePixelValue)).map(b -> b ? 0 : 1).collect(Collectors.toList());;
		
		// Find the sum of values multiplied by the corresponding power of 2
		Double sum = 0d;
		for(int i = 0; i < nbhList.size(); i++){
			sum = sum + nbhBinList.get(i) * (pow(2, i));
		}
		
		return sum;
	}

}
