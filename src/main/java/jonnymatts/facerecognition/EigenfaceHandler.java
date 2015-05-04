package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class EigenfaceHandler {

	private Mat blueChannelMeanImage;
	private Mat greenChannelMeanImage;
	private Mat redChannelMeanImage;
	private Mat depthMeanImage;
	private Mat blueEigenfaceMat;
	private Mat greenEigenfaceMat;
	private Mat redEigenfaceMat;
	private Mat depthEigenfaceMat;
	private List<Mat> colourTrainingImageList;
	private List<Mat> depthTrainingImageList;
	private List<List<Mat>> colourEigenfaceList;
	private List<Mat> depthEigenfaceList;
	
	public static String getExtractorName() {
		return "_EIG";
	}

	public EigenfaceHandler(List<Mat> cti, List<Mat> dti) {
		colourTrainingImageList = cti;
		depthTrainingImageList = dti;
		colourEigenfaceList = createEigenfaces();
		depthEigenfaceList = createDepthEigenfaces();
	}

	public void displayDepthEigenfaces() {
		displayImageList(depthEigenfaceList);
	}

	public void displayEigenfacesForColourChannel(int channel) {
		displayImageList(colourEigenfaceList.get(channel));
	}

	List<List<Double>> findFeatureVector(Mat colourImage, Mat depthImage) {
		List<List<Double>> featureVector = new ArrayList<List<Double>>();
		
		// Find weights for colour and depth image, then concatenate them
		List<List<Double>> colourWeights = calculateWeightsForGivenColourImage(colourImage);
		List<Double> depthWeights = calculateWeightsForGivenDepthImage(depthImage);
		
		for(List<Double> weights : colourWeights) {
			featureVector.add(weights);
		}
		featureVector.add(depthWeights);
		
		return featureVector;
	}

	List<Double> calculateWeightsForGivenDepthImage(Mat depthImage) {
		
		// Normalise depth images, then convert to greyscale
		Mat normalisedDepthImage = normaliseDepthImage(depthImage);
		normalisedDepthImage.convertTo(normalisedDepthImage, CvType.CV_8UC1);

		// Subtract mean image
		Mat meanSubtractedImage = subtractSingleChannelImages(normalisedDepthImage,
				depthMeanImage);

		Mat depthEigenfaceWeights = new Mat();

		// Multiply mean subtracted image by transpose of eigenface matrix
		Core.gemm(convertImageToSingleRowMatrix(meanSubtractedImage, 0),
				depthEigenfaceMat.t(), 1, new Mat(), 0, depthEigenfaceWeights,
				0);

		return convertMatToList(depthEigenfaceWeights, 0);
	}

	List<List<Double>> calculateWeightsForGivenColourImage(Mat img) {

		// Subtract mean image of each colour channel from image
		Mat meanSubtractedImage = subtractImageFromMultiChannelImage(img,
				blueChannelMeanImage, greenChannelMeanImage,
				redChannelMeanImage);

		Mat blueEigenfaceWeights = new Mat();
		Mat greenEigenfaceWeights = new Mat();
		Mat redEigenfaceWeights = new Mat();

		// Multiply mean subtracted image by transpose of eigenface matrix
		Core.gemm(convertImageToSingleRowMatrix(meanSubtractedImage, 0),
				blueEigenfaceMat.t(), 1, new Mat(), 0, blueEigenfaceWeights, 0);

		Core.gemm(convertImageToSingleRowMatrix(meanSubtractedImage, 1),
				greenEigenfaceMat.t(), 1, new Mat(), 0, greenEigenfaceWeights,
				0);

		Core.gemm(convertImageToSingleRowMatrix(meanSubtractedImage, 2),
				redEigenfaceMat.t(), 1, new Mat(), 0, redEigenfaceWeights, 0);

		return Arrays.asList(convertMatToList(blueEigenfaceWeights, 0),
				convertMatToList(greenEigenfaceWeights, 0),
				convertMatToList(redEigenfaceWeights, 0));
	}

	List<Mat> findEigenvectorsForMatrix(Mat in) {
		// Mutliply each A-Matrix by it's transpose, then find eigenvectors of
		// inner product matrix
		Mat lMatrix = new Mat();
		Mat innerProductEigenValues = new Mat();
		Mat innerProductEigenVectors = new Mat();
		Mat eigenVectors = new Mat();
		List<Mat> returnList = new ArrayList<Mat>();

		Core.gemm(in, in.t(), 1, new Mat(), 0, lMatrix, 0);

		// Find eigenvectors and eigenvalues
		Core.eigen(lMatrix, true, innerProductEigenValues,
				innerProductEigenVectors);

		Core.gemm(innerProductEigenVectors, in, 1, new Mat(), 0, eigenVectors,
				0);

		for (int im = 0; im < colourTrainingImageList.size(); im++) {
			Mat eigenfaceImage = new Mat(colourTrainingImageList.get(0).rows(),
					colourTrainingImageList.get(0).cols(), CvType.CV_64FC1);
			int dataIndex = 0;
			for (int i = 0; i < colourTrainingImageList.get(0).cols(); i++) {
				for (int j = 0; j < colourTrainingImageList.get(0).rows(); j++) {
					eigenfaceImage.put(i, j, eigenVectors.get(im, dataIndex));
					dataIndex++;
				}
			}
			returnList.add(eigenfaceImage);
		}

		return returnList;
	}

	List<Mat> createDepthEigenfaces() {
		Mat aMatrix = new Mat(depthTrainingImageList.size(),
				(int) depthTrainingImageList.get(0).size().area(),
				CvType.CV_64F);
		int depthImageRows = depthTrainingImageList.get(0).rows();
		int depthImageCols = depthTrainingImageList.get(0).cols();
		Mat meanMatrix = new Mat(depthImageRows, depthImageCols,
				CvType.CV_64FC1);

		// Normalise depth images, then convert to greyscale
		List<Mat> greyscaleDepthTrainingImageList = depthTrainingImageList
				.stream().map(i -> {
					Mat image = normaliseDepthImage(i);
					image.convertTo(image, CvType.CV_8UC1);
					return image;
				}).collect(Collectors.toList());

		// Find mean matrix
		for (int i = 0; i < depthImageCols; i++) {
			for (int j = 0; j < depthImageRows; j++) {
				int sum = 0;
				for (int n = 0; n < greyscaleDepthTrainingImageList.size(); n++) {
					sum += greyscaleDepthTrainingImageList.get(n).get(j, i)[0];
				}
				meanMatrix.put(j, i,
						(sum / greyscaleDepthTrainingImageList.size()));
			}
		}

		// Save depth channel image for use later
		depthMeanImage = meanMatrix;

		// Find the mean subtracted value for each image and channel
		List<Mat> meanSubtractedTrainingImageList = new ArrayList<Mat>();
		for (Mat im : greyscaleDepthTrainingImageList) {
			Mat meanSubtractedImage = im.clone();
			for (int i = 0; i < depthImageCols; i++) {
				for (int j = 0; j < depthImageRows; j++) {
					double val = meanSubtractedImage.get(j, i)[0];
					double newVal = (val - meanMatrix.get(j, i)[0]);
					meanSubtractedImage.put(j, i, newVal);
				}
			}
			meanSubtractedTrainingImageList.add(meanSubtractedImage);
		}

		int rowIndex = 0;
		// Create A matrix containing all images, one per row
		for (Mat im : meanSubtractedTrainingImageList) {
			List<Double> imList = convertMatToList(im, 0);
			for (int i = 0; i < imList.size(); i++) {
				aMatrix.put(rowIndex, i, imList.get(i));
			}
			rowIndex++;
		}
		
		List<Mat> depthEigenList = findEigenvectorsForMatrix(aMatrix);
		int imageSize = (int) depthEigenList.get(0).size().area();
		depthEigenfaceMat = convertImageListToSingleMatrix(depthEigenList, depthEigenList.size(), imageSize);

		return depthEigenList;
	}

	List<List<Mat>> createEigenfaces() {
		Mat bAMatrix = new Mat(colourTrainingImageList.size(),
				(int) colourTrainingImageList.get(0).size().area(),
				CvType.CV_64F);
		Mat gAMatrix = new Mat(colourTrainingImageList.size(),
				(int) colourTrainingImageList.get(0).size().area(),
				CvType.CV_64F);
		Mat rAMatrix = new Mat(colourTrainingImageList.size(),
				(int) colourTrainingImageList.get(0).size().area(),
				CvType.CV_64F);
		Mat bMeanMatrix = new Mat(colourTrainingImageList.get(0).rows(),
				colourTrainingImageList.get(0).cols(), CvType.CV_64FC1);
		Mat gMeanMatrix = new Mat(colourTrainingImageList.get(0).rows(),
				colourTrainingImageList.get(0).cols(), CvType.CV_64FC1);
		Mat rMeanMatrix = new Mat(colourTrainingImageList.get(0).rows(),
				colourTrainingImageList.get(0).cols(), CvType.CV_64FC1);

		// Find mean matrix for each colour channel
		for (int i = 0; i < colourTrainingImageList.get(0).cols(); i++) {
			for (int j = 0; j < colourTrainingImageList.get(0).rows(); j++) {
				int ySum = 0;
				int uSum = 0;
				int vSum = 0;
				for (int n = 0; n < colourTrainingImageList.size(); n++) {
					ySum += colourTrainingImageList.get(n).get(j, i)[0];
					uSum += colourTrainingImageList.get(n).get(j, i)[1];
					vSum += colourTrainingImageList.get(n).get(j, i)[2];
				}
				bMeanMatrix.put(j, i, (ySum / colourTrainingImageList.size()));
				gMeanMatrix.put(j, i, (uSum / colourTrainingImageList.size()));
				rMeanMatrix.put(j, i, (vSum / colourTrainingImageList.size()));
			}
		}

		// Save the mean images for use later
		blueChannelMeanImage = bMeanMatrix;
		greenChannelMeanImage = gMeanMatrix;
		redChannelMeanImage = rMeanMatrix;

		// Find the mean subtracted value for each image and channel
		List<Mat> meanSubtractedTrainingImageList = new ArrayList<Mat>();
		for (Mat im : colourTrainingImageList) {
			Mat meanSubtractedImage = im.clone();
			for (int i = 0; i < colourTrainingImageList.get(0).cols(); i++) {
				for (int j = 0; j < colourTrainingImageList.get(0).rows(); j++) {
					double[] vals = meanSubtractedImage.get(j, i);
					double[] newVals = { (vals[0] - bMeanMatrix.get(j, i)[0]),
							(vals[1] - gMeanMatrix.get(j, i)[0]),
							(vals[2] - rMeanMatrix.get(j, i)[0]) };
					meanSubtractedImage.put(j, i, newVals);
				}
			}
			meanSubtractedTrainingImageList.add(meanSubtractedImage);
		}

		int rowIndex = 0;
		// Create A matrix containing all images, one per row
		for (Mat im : meanSubtractedTrainingImageList) {
			List<Double> bImList = convertMatToList(im, 0);
			List<Double> gImList = convertMatToList(im, 1);
			List<Double> rImList = convertMatToList(im, 2);
			for (int i = 0; i < bImList.size(); i++) {
				bAMatrix.put(rowIndex, i, bImList.get(i));
				gAMatrix.put(rowIndex, i, gImList.get(i));
				rAMatrix.put(rowIndex, i, rImList.get(i));
			}
			rowIndex++;
		}

		// Find eigenvectors for each colour channel
		List<List<Mat>> colourChannelEigenVectors = Arrays.asList(
				findEigenvectorsForMatrix(bAMatrix),
				findEigenvectorsForMatrix(gAMatrix),
				findEigenvectorsForMatrix(rAMatrix));
		
		// Create eigenface matrix for each colour channel
		int imageSize = (int) colourChannelEigenVectors.get(0).get(0).size().area();
		blueEigenfaceMat = convertImageListToSingleMatrix(
				colourChannelEigenVectors.get(0), colourChannelEigenVectors.get(0).size(), imageSize);
		greenEigenfaceMat = convertImageListToSingleMatrix(
				colourChannelEigenVectors.get(1), colourChannelEigenVectors.get(0).size(), imageSize);
		redEigenfaceMat = convertImageListToSingleMatrix(colourChannelEigenVectors.get(2),
				colourChannelEigenVectors.get(0).size(), imageSize);

		return colourChannelEigenVectors;
	}
}
