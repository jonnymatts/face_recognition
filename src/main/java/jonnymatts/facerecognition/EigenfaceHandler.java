package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class EigenfaceHandler {

	private List<Mat> trainingImageList;
	private List<List<Mat>> eigenfaceList;
	
	public EigenfaceHandler(List<Mat> ti) {
		trainingImageList = ti;
		eigenfaceList = createEigenfaces();
	}
	
	public void displayEigenfacesForChannel(int channel) {
		for(Mat im : eigenfaceList.get(channel)) {
			displayImage(im);
		}
	}
	
	List<Mat> findEigenvectorsForMatrix(Mat in) {
		// Mutliply each A-Matrix by it's transpose, then find eigenvectors of inner product matrix
		Mat lMatrix = new Mat();
		Mat innerProductEigenValues = new Mat();
		Mat innerProductEigenVectors = new Mat();
		Mat eigenVectors = new Mat();
		List<Mat> returnList = new ArrayList<Mat>();

		Core.gemm(in, in.t(), 1, new Mat(), 0, lMatrix, 0);

		Core.eigen(lMatrix, true, innerProductEigenValues,
				innerProductEigenVectors);

		Core.gemm(innerProductEigenVectors, in, 1, new Mat(), 0,
				eigenVectors, 0);

		for (int im = 0; im < trainingImageList.size(); im++) {
			Mat eigenfaceImage = new Mat(trainingImageList.get(0).rows(),
					trainingImageList.get(0).cols(), CvType.CV_64FC1);
			int dataIndex = 0;
			for (int i = 0; i < trainingImageList.get(0).cols(); i++) {
				for (int j = 0; j < trainingImageList.get(0).rows(); j++) {
					eigenfaceImage.put(i, j, eigenVectors.get(im, dataIndex));
					dataIndex++;
				}
			}
			returnList.add(eigenfaceImage);
		}

		return returnList;
	}
	
	List<List<Mat>> createEigenfaces() {
		Mat bAMatrix = new Mat(trainingImageList.size(), (int)trainingImageList.get(0).size().area(), CvType.CV_64F);
		Mat gAMatrix = new Mat(trainingImageList.size(), (int)trainingImageList.get(0).size().area(), CvType.CV_64F);
		Mat rAMatrix = new Mat(trainingImageList.size(), (int)trainingImageList.get(0).size().area(), CvType.CV_64F);
		Mat bMeanMatrix = new Mat(trainingImageList.get(0).rows(), trainingImageList.get(0).cols(), CvType.CV_64FC1);
		Mat gMeanMatrix = new Mat(trainingImageList.get(0).rows(), trainingImageList.get(0).cols(), CvType.CV_64FC1);
		Mat rMeanMatrix = new Mat(trainingImageList.get(0).rows(), trainingImageList.get(0).cols(), CvType.CV_64FC1);
		
		// Find mean matrix for each colour channel
		for(int i = 0; i < trainingImageList.get(0).cols(); i++) {
			for(int j = 0; j < trainingImageList.get(0).rows(); j++) {
				int ySum = 0; int uSum = 0; int vSum = 0;
				for(int n = 0; n < trainingImageList.size(); n++) {
					ySum += trainingImageList.get(n).get(j, i)[0];
					uSum += trainingImageList.get(n).get(j, i)[1];
					vSum += trainingImageList.get(n).get(j, i)[2];
				}
				bMeanMatrix.put(j, i, (ySum / trainingImageList.size()));
				gMeanMatrix.put(j, i, (uSum / trainingImageList.size()));
				rMeanMatrix.put(j, i, (vSum / trainingImageList.size()));
			}
		}
		
		// Find the mean subtracted value for each image and channel
		List<Mat> meanSubtractedTrainingImageList = new ArrayList<Mat>();
		for(Mat im : trainingImageList) {
			Mat meanSubtractedImage = im.clone();
			for(int i = 0; i < trainingImageList.get(0).cols(); i++) {
				for(int j = 0; j < trainingImageList.get(0).rows(); j++) {
					double[] vals = meanSubtractedImage.get(j, i);
					double[] newVals = {(vals[0] - bMeanMatrix.get(j, i)[0]), (vals[1] - gMeanMatrix.get(j, i)[0]),
										(vals[2] - rMeanMatrix.get(j, i)[0])};
					meanSubtractedImage.put(j, i, newVals);
				}
			}
			meanSubtractedTrainingImageList.add(meanSubtractedImage);
		}
		
		int rowIndex = 0;
		// Create A matrix containing all images, one per row
		for(Mat im : meanSubtractedTrainingImageList) {
			List<Double> yImList = convertMatToList(im, 0);
			List<Double> uImList = convertMatToList(im, 1);
			List<Double> vImList = convertMatToList(im, 2);
			for(int i = 0; i < yImList.size(); i++) {
				bAMatrix.put(rowIndex, i, yImList.get(i));
				gAMatrix.put(rowIndex, i, uImList.get(i));
				rAMatrix.put(rowIndex, i, vImList.get(i));
			}
			rowIndex++;
		}
		
		// Find eigenvectors for each colour channel
		List<List<Mat>> colourChannelEigenVectors = Arrays.asList(findEigenvectorsForMatrix(bAMatrix),
																  findEigenvectorsForMatrix(gAMatrix),
																  findEigenvectorsForMatrix(rAMatrix));
		
		return colourChannelEigenVectors;
	}
}
