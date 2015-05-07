package jonnymatts.facerecognition;

import static java.lang.Math.*;
import static jonnymatts.facerecognition.ApplicationUtil.flattenList;
import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import com.google.common.primitives.Ints;

public class LocalBinaryPatternHandler implements ProjectFeatureExtractor {

	private boolean useUniformPatterns;
	private boolean useRotationInvariance;
	private boolean useRGB;
	private int population;
	private int radius;
	private List<Double> histogramBinList;
	private int noOfSubImages;
	
	public String getExtractorName() {
		String returnString = "_LBP_" + population + "_" + radius + "_" + noOfSubImages;
		if(useUniformPatterns) returnString += "_UP";
		if(useRotationInvariance) returnString += "_RI";
		if(useRGB) returnString += "_RGB";
		return returnString;
	}

	public void setPopulation(int p) {
		population = p;
	}

	public int getPopulation() {
		return population;
	}

	public void setRadius(int r) {
		radius = r;
	}

	public int getRadius() {
		return radius;
	}

	public void setUseUniformPatterns(boolean b) {
		useUniformPatterns = b;
		histogramBinList = createHistogramBinList();
	}
	
	public void setUseRotationInvariance(boolean b) {
		useRotationInvariance = b;
	}
	
	public void setUseRGB(boolean b) {
		useRGB = b;
	}

	public LocalBinaryPatternHandler(int p, int r, boolean u, boolean ri, boolean rgb, int n) {
		population = p;
		radius = r;
		useUniformPatterns = u;
		useRotationInvariance = ri;
		useRGB = rgb;
		noOfSubImages = n;
		histogramBinList = createHistogramBinList();
	}
	
	List<Integer> getUniformPatternList(List<List<Integer>> intLists, List<Integer> intList) {
		List<Boolean> boolList = intLists.stream().map(b -> {
			int noOfChanges = 0;
			for (int i = 1; i < b.size(); i++) {
				if (b.get(i) != b.get(i - 1))
					noOfChanges++;
			}
			return noOfChanges;
		}).map(s -> s < 3).collect(Collectors.toList());
		return intList.stream().filter(i -> boolList.get(i)).collect(Collectors.toList());
	}

	private List<Double> createHistogramBinList() {
		List<Integer> intList = IntStream.range(0, (int) pow(2, population)).boxed().collect(Collectors.toList());
		List<List<Integer>> intLists = intList.stream().map(i -> convertIntegerToBinaryList(i)).collect(Collectors.toList());
		List<Double> returnList;
		
		if (useUniformPatterns && useRotationInvariance) {
			List<Integer> uniformPatternList = getUniformPatternList(intLists, intList);
			returnList = uniformPatternList.stream().map(b -> findRotationInvariantSequence(convertIntegerToBinaryList(b))).distinct().map(b -> convertBinaryListToInteger(b).doubleValue()).collect(Collectors.toList());
			
		} else if (useUniformPatterns) {
			returnList = getUniformPatternList(intLists, intList).stream().map(i -> i.doubleValue()).collect(Collectors.toList());
			
		} else if (useRotationInvariance) {
			returnList = intLists.stream().map(b -> findRotationInvariantSequence(b)).distinct().map(b -> convertBinaryListToInteger(b).doubleValue()).collect(Collectors.toList());
			
		} else {
			returnList = intList.stream().map(i -> i.doubleValue()).collect(Collectors.toList());
		}
		
		return new ArrayList<Double>(returnList.stream().sorted().collect(Collectors.toList()));
	}
	
	// Finds the orientation of bits within the list that give the maxiumum value
	List<Integer> findRotationInvariantSequence(List<Integer> in) {
		
		// Set the return variables
		int highestVal = convertBinaryListToInteger(in);
		List<Integer> returnList = in;
		List<Integer> currentList = in;
		
		// Shift bits of array to the left and calculate the value 
		for(int i = 1; i < in.size()+1; i++) {
			List<Integer> newList = shiftBinaryList(currentList);
			int newVal = convertBinaryListToInteger(in);
			if(newVal > highestVal) {
				highestVal = newVal;
				returnList = new ArrayList<Integer>(newList);
			}
			currentList = newList;
		}
		return returnList;
	}
	
	List<Integer> shiftBinaryList(List<Integer> in) {
		
		// Retrieve head of list
		int head = in.get(0);
		
		// Shift each list value to the left once
		for(int i = 1; i < in.size(); i++) {
			in.set((i-1), in.get(i));
		}
		
		// Replace head value at the end of the list
		in.set((in.size()-1), head);
		return in;
	}

	List<Integer> convertIntegerToBinaryList(int i) {
		int[] binList = new int[population];
		for (int j = 0; j < population; ++j) {
			binList[population - j - 1] = i & 0x1;
			i >>= 1;
		}
		return Ints.asList(binList);
	}

	Integer convertBinaryListToInteger(List<Integer> bList) {
		Integer sum = 0;
		for (int i = 0; i < bList.size(); i++) {
			sum = sum + (int) (bList.get(i) * pow(2, (bList.size()-1)-i));
		}
		return sum;
	}
	
	private List<Double> convertImageToHistogram(Mat img) {
		// Choose correct size for output histogram
		int histSize;
		if (useUniformPatterns && useRotationInvariance) {
			histSize = histogramBinList.size() + 1;
		} else if (useUniformPatterns) {
			histSize = histogramBinList.size() + 1;
		} else if (useRotationInvariance) {
			histSize = histogramBinList.size() + 1;
		} else {
			histSize = (int) pow(2, (double) population);
		}

		// Create histogram list for sub-image
		Double[] histArr = new Double[histSize];

		// Fill histogram with default values (0)
		Arrays.fill(histArr, 0d);
		List<Double> imgHist = Arrays.asList(histArr);

		List<Double> lbpValueList = convertMatToList(img, 0);

		// Limit values between 0 and maxValue
		lbpValueList = lbpValueList.stream().map(v -> (double)round(max(0, min(histSize - 1, v)))).collect(Collectors.toList());
		
		// Populate histogram with LBP data
		for (Double val : lbpValueList) {
			Integer iVal = val.intValue();
			if (useUniformPatterns || useRotationInvariance) {
				if (histogramBinList.contains(val)) { 
					imgHist.set(iVal, (imgHist.get(iVal) + 1));
				} else {
					int lastIndex = imgHist.size() - 1;
					imgHist.set(lastIndex, (imgHist.get(lastIndex) + 1));
				}
			} else {
				imgHist.set(iVal, (imgHist.get(iVal) + 1));
			}
		}

		return imgHist;
	}

	// Find the feature vector for given image, population and radius
	public List<List<Double>> findFeatureVector(Mat img) {
		
		if(!useRGB) img.convertTo(img, CvType.CV_8U);
		
		// Make sure uniformPatternList is correct
		histogramBinList = createHistogramBinList();

		// Split image into grid of sub-images
		List<Mat> subImgList = new ArrayList<Mat>();
		int subImgWidth = (int) floor(img.cols() / noOfSubImages);
		int subImgHeight = (int) floor(img.rows() / noOfSubImages);

		for (int i = 0; i < noOfSubImages; i++) {
			for (int j = 0; j < noOfSubImages; j++) {
				
				// Find x and y values for sub-image vertices
				int iIndex = i * subImgWidth;
				int jIndex = j * subImgHeight;
				int iIndex2 = min((iIndex + subImgWidth), img.cols());
				int jIndex2 = min((jIndex + subImgHeight), img.rows());
				subImgList.add(img.submat(jIndex, jIndex2, iIndex, iIndex2));
			}
		}

		// Create list for feature vector
		List<List<Double>> histList = new ArrayList<List<Double>>();
		
		for (Mat subImg : subImgList) {

			if(!useRGB) {
				// Find the LBP value at each pixel
				Mat lbpImg = calculateLBP(subImg, LBPColour.GREY);
				
				// Convert each sub-image into a histogram
				List<Double> imgHist = convertImageToHistogram(lbpImg);
				
				histList.add(imgHist);
			} else {
				List<Mat> bgrList = new ArrayList<Mat>();
				Core.split(subImg, bgrList);
				for(Mat channelImage : bgrList) {
					
					// Find the LBP value at each pixel
					Mat lbpImg = calculateLBP(channelImage, LBPColour.BLUE);
					
					// Convert each sub-image into a histogram
					List<Double> imgHist = convertImageToHistogram(lbpImg);
					
					histList.add(imgHist);
				}
			}
		}
		
		return histList;
	}

	// Calculate the LBP for each pixel within the image
	Mat calculateLBP(Mat img, LBPColour lbpColour) {
		Mat lbpImg = new Mat(img.rows(), img.cols(), CvType.CV_64F);

		for (int i = radius; i < (img.cols() - radius); i++) {
			for (int j = radius; j < (img.rows() - radius); j++) {
				if ((i < radius) || (i > img.cols() - radius) || (j < radius)
						|| (j > img.rows() - radius)) {
					lbpImg.put(j, i, 0);
				}
				lbpImg.put(j, i, calculateLBPForPixel(img, i, j, lbpColour));
			}
		}

		return lbpImg;
	}

	// Calculate the LBP for each pixel within the image
	Double calculateLBPForPixel(Mat img, int x, int y, LBPColour lbpColour) {
		List<Double> nbhList = new ArrayList<Double>();

		// Use linear interpolation to find circular LBP values
		double anglePortion = (2 * PI) / population;
		for (int i = 0; i < population; i++) {
			double angle = i * anglePortion;
			double px = x + (radius * cos(angle));
			double py = y + (radius * sin(angle));
			
			// Find values of the 4 surrounding cells
			double xMin = floor(px);
			int xMinIndex = (int) xMin;
			double xMax = ceil(px);
			int xMaxIndex = (int) xMax;
			double yMin = floor(py);
			int yMinIndex = (int) yMin;
			double yMax = ceil(py);
			int yMaxIndex = (int) yMax;
			
			double x1, x2, x3, x4;
			if(!useRGB) {
				x1 = img.get(yMinIndex, xMinIndex)[0];
				x2 = img.get(yMinIndex, xMaxIndex)[0];
				x3 = img.get(yMaxIndex, xMinIndex)[0];
				x4 = img.get(yMaxIndex, xMaxIndex)[0];
			} else {
				x1 = img.get(yMinIndex, xMinIndex)[lbpColour.getValue()];
				x2 = img.get(yMinIndex, xMaxIndex)[lbpColour.getValue()];
				x3 = img.get(yMaxIndex, xMinIndex)[lbpColour.getValue()];
				x4 = img.get(yMaxIndex, xMaxIndex)[lbpColour.getValue()];
			}
			
			// Use bilinear interpolation if needed
			double value = ((px == 0) || (py == 0) || ((px % 1 == 0) && (py % 1 == 0))) ? img.get((int) round(py), (int) round(px))[0] : bilinearInterpolation(px, py, xMin, xMax, yMin, yMax, x1, x2, x3, x4);
			nbhList.add(value);
		}

		// Remove and preserve central pixel value from list
		Double centrePixelValue = img.get(y, x)[0];

		// Convert list value to either 1 or 0
		// if(value < centrePixelValue) 0 else 1
		List<Integer> nbhBinList = nbhList.stream().map(d -> (d < centrePixelValue)).map(b -> b ? 0 : 1).collect(Collectors.toList());
		
		return convertBinaryListToInteger(nbhBinList).doubleValue();
	}

	public PersonDataset performFeatureExtractionForDataset(PersonDataset ds, boolean sample) {
		List<Person> personList = ds.getPersonList();
		for (Person p : personList) {
			List<List<Double>> featureVector = findFeatureVector(p.colourImage);
			List<Double> fv = flattenList(featureVector);
			if(sample) fv = sampleFeatureVector(fv);
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}
}
