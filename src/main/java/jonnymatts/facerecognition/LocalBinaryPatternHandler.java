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

import com.google.common.primitives.Ints;

public class LocalBinaryPatternHandler {

	private boolean useUniformPatterns;
	private int population;
	private int radius;
	private List<Double> uniformPatternList;

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
		uniformPatternList = createUniformPatternList();
	}

	public LocalBinaryPatternHandler(int p, int r, boolean u) {
		population = p;
		radius = r;
		useUniformPatterns = u;
		uniformPatternList = createUniformPatternList();
	}

	private List<Double> createUniformPatternList() {
		List<Integer> intList = IntStream.range(0, (int) pow(2, population))
				.boxed().collect(Collectors.toList());
		List<List<Integer>> intLists = intList.stream()
				.map(i -> convertIntegerToBinaryList(i))
				.collect(Collectors.toList());
		List<Boolean> boolList = intLists.stream().map(b -> {
			int noOfChanges = 0;
			for (int i = 1; i < b.size(); i++) {
				if (b.get(i) != b.get(i - 1))
					noOfChanges++;
			}
			return noOfChanges;
		}).map(s -> s < 3).collect(Collectors.toList());
		return useUniformPatterns ? intList.stream()
				.filter(i -> boolList.get(i)).map(i -> i.doubleValue())
				.collect(Collectors.toList()) : null;
	}

	List<Integer> convertIntegerToBinaryList(int i) {
		int[] digits = new int[population];
		for (int j = 0; j < population; ++j) {
			digits[population - j - 1] = i & 0x1;
			i >>= 1;
		}
		return Ints.asList(digits);
	}

	Integer convertBinaryListToIntger(List<Integer> bList) {
		Integer sum = 0;
		for (int i = 0; i < population; i++) {
			sum = sum + (int) (bList.get(i) * pow(2, i));
		}
		return sum;
	}

	// Find the feature vector for given image, population and radius
	public List<List<Double>> findFeatureVector(Mat img, int noOfSubImgs) {

		// Split image into grid of sub-images
		List<Mat> subImgList = new ArrayList<Mat>();
		int subImgWidth = (int) floor(img.cols() / noOfSubImgs);
		int subImgHeight = (int) floor(img.rows() / noOfSubImgs);

		for (int i = 0; i < noOfSubImgs; i++) {
			for (int j = 0; j < noOfSubImgs; j++) {
				int iIndex = i * subImgWidth;
				int jIndex = j * subImgHeight;
				subImgList.add(img.submat(iIndex, iIndex + subImgWidth, jIndex,
						jIndex + subImgHeight));
			}
		}

		// Find histogram for each sub-image
		List<List<Double>> histList = new ArrayList<List<Double>>();
		for (Mat subImg : subImgList) {

			// Create histogram list for sub-image
			int histSize = useUniformPatterns ? (uniformPatternList.size() + 1)
					: (int) pow(2, (double) population);
			Double[] histArr = new Double[histSize];

			// Fill histogram with default values (0)
			Arrays.fill(histArr, 0d);
			List<Double> imgHist = Arrays.asList(histArr);
			Collections.fill(imgHist, 0d);

			// Find the LBP value at each pixel
			Mat lbpImg = calculateLBP(subImg);

			List<Double> lbpValueList = convertMatToList(lbpImg);

			// Limit values between 0 and maxValue
			lbpValueList = lbpValueList.stream()
					.map(v -> max(0, min(histSize - 1, v)))
					.collect(Collectors.toList());

			// Populate histogram with LBP data
			for (Double val : lbpValueList) {
				Integer iVal = val.intValue();
				if (useUniformPatterns) {
					if (uniformPatternList.contains(iVal))
						imgHist.set(iVal, (imgHist.get(iVal) + 1));
					imgHist.stream().filter(i -> i != 0)
							.collect(Collectors.toList());
				} else {
					imgHist.set(iVal, (imgHist.get(iVal) + 1));
				}
			}

			histList.add(imgHist);
		}

		return histList;
	}

	// Calculate the LBP for each pixel within the image
	Mat calculateLBP(Mat img) {
		Mat lbpImg = new Mat(img.rows(), img.cols(), CvType.CV_64F);

		for (int i = radius; i < (img.cols() - radius); i++) {
			for (int j = radius; j < (img.rows() - radius); j++) {
				if ((i < radius) || (i > img.cols() - radius) || (j < radius)
						|| (j > img.rows() - radius)) {
					lbpImg.put(j, i, 0);
				}
				lbpImg.put(j, i, calculateLBPForPixel(img, i, j));
			}
		}

		return lbpImg;
	}

	// Calculate the LBP for each pixel within the image
	Double calculateLBPForPixel(Mat img, int x, int y) {
		List<Double> nbhList = new ArrayList<Double>();

		// Use linear interpolation to find circular LBP values
		double anglePortion = (2 * PI) / population;
		for (int i = 0; i < population; i++) {
			double angle = i * anglePortion;
			double px = x + (radius * cos(angle));
			double py = y + (radius * sin(angle));
			double value = ((px == 0) || (py == 0) || ((px % 1 == 0) && (py % 1 == 0))) ? img
					.get((int) round(py), (int) round(px))[0]
					: bilinearInterpolation(img, px, py);
			nbhList.add(value);
		}

		// Remove and preserve central pixel value from list
		Double centrePixelValue = img.get(y, x)[0];

		// Convert list value to either 1 or 0
		// if(value < centrePixelValue) 0 else 1
		List<Integer> nbhBinList = nbhList.stream()
				.map(d -> (d < centrePixelValue)).map(b -> b ? 0 : 1)
				.collect(Collectors.toList());
		;

		// Find the sum of values multiplied by the corresponding power of 2
		Double sum = 0d;
		for (int i = 0; i < nbhList.size(); i++) {
			sum = sum + nbhBinList.get(i) * (pow(2, i));
		}

		return sum;
	}

}
