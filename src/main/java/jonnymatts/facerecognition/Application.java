package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.*;
import static jonnymatts.facerecognition.ImageHelper.*;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.google.common.primitives.Ints;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");

		// // Read in result sets
		// PersonDataset set1 =
		// readResultSet("/resources/results/EURECOM_1_testing_LBP_8_1_5_UP_RGB_KNN_3.txt");
		// PersonDataset set2 =
		// readResultSet("/resources/results/EURECOM_2_testing_LBP_8_1_5_UP_RGB_KNN_3.txt");
		// PersonDataset set3 =
		// readResultSet("/resources/results/EURECOM_3_testing_LBP_8_1_5_UP_RGB_KNN_3.txt");
		// PersonDataset set4 =
		// readResultSet("/resources/results/EURECOM_4_testing_LBP_8_1_5_UP_RGB_KNN_3.txt");
		// PersonDataset set5 =
		// readResultSet("/resources/results/EURECOM_5_testing_LBP_8_1_5_UP_RGB_KNN_3.txt");
		//
		// DatasetResult results1 = set1.getDatasetResult();
		// DatasetResult results2 = set2.getDatasetResult();
		// DatasetResult results3 = set3.getDatasetResult();
		// DatasetResult results4 = set4.getDatasetResult();
		// DatasetResult results5 = set5.getDatasetResult();
		// DatasetResult average =
		// averageDatasetResultList(Arrays.asList(results1, results2, results3,
		// results4, results5));
		//
		// System.out.println("results1: " results1.getResultString());
		// System.out.println("results2: " results2.getResultString());
		// System.out.println("results3: " results3.getResultString());
		// System.out.println("results4: " results4.getResultString());
		// System.out.println("results5: " results5.getResultString());
		// System.out.println("average: " average.getResultString());

		// Path start = FileSystems.getDefault().getPath(userDir
		// "/resources/face_testing_images/IIIT-D Kinect RGB-D Face Database/fold1/testing");
		// List<String> stringList = Files.walk(start)
		// .filter(path -> path.toFile().isFile())
		// .filter(path -> path.toString().endsWith(".jpg"))
		// .map(path -> path.toString())
		// .collect(Collectors.toList());
		//
		// System.out.println(stringList.size());

//		PersonDataset set = readDataset("/resources/datasets/IIT-D_fold1_testing.txt");
//		PersonDatasetAnalytics pda = new PersonDatasetAnalytics(set);
//		pda.addPeopleToPersonList(readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting1.txt"));
//		List<PersonDataset> sets = pda.getOptimizedDatasets("IIT-D_EURECOM_combiation_1", 100);
//		PersonDataset trainingSet = sets.get(0);
//		PersonDataset testingSet = sets.get(1);
//		
//		PersonDatasetAnalytics trainPda = new PersonDatasetAnalytics(trainingSet);
//		PersonDatasetAnalytics testPda = new PersonDatasetAnalytics(testingSet);
		
		PersonDataset set1 = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting1.txt");
		PersonDataset set2 = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting2.txt");
		PersonDataset set3 = readDataset("/resources/datasets/IIT-D_fold1_testing.txt");
		
		set1.setPersonList(performPreprocessing(set1, 256));
		
		Mat image = set1.getPersonList().get(42).colourImage;
		Mat depthImage = set1.getPersonList().get(42).depthImage;
		
//		displayImage(image);
//		displayImage(preprocessImages(image, image, 128).get(0));
		
		GradientLBPHandler glbph = new GradientLBPHandler(8, 1);
		List<List<Double>> featureVector = glbph.findFeatureVector(image, depthImage);
//		Mat salImage = readImageFromFile("/resources/me.jpg");
//		Imgproc.resize(salImage, salImage, new Size(256, 256));
//		displayImage(rh.findSaliencyMap(salImage));
//		
//		List<Integer> intList = Arrays.asList(1);
//		for (Integer i : intList) {
//			System.out.println("[" + i + "] Starting image pre-processing...");
//
//			// Pre-process all the images
//			trainingSet.setPersonList(performPreprocessing(trainingSet, 256));
//			testingSet.setPersonList(performPreprocessing(testingSet, 256));
//
//			// Perform feature extraction on both datasets
//
//			System.out.println("[" + i + "] Starting feature extraction on training set...");
//
//			performLBPFeatureExtractionOnDataset(trainingSet, 5, 8, 1, false, false, false);
//			
//			PersonDataset eTrainingSet = performLBPFeatureExtractionOnDataset(trainingSet, 5, 8, 1, false, false, false);
//
//			System.out.println("[" + i + "] Starting feature extraction on testing set...");
//
//			PersonDataset eTestingSet = performLBPFeatureExtractionOnDataset(testingSet, 5, 8, 1, false, false, false);
//
//			System.out.println("[" + i + "] Starting gender biometric prediction...");
//
//			// Train the classifer, then predict the class of testing data for
//			// each biometric
//			SVMHandler classifier = new SVMHandler();
//			classifier.trainForBiometric(eTrainingSet, Biometric.GENDER);
//			PersonDataset resultSet = classifier.predictClassesForBiometric(eTestingSet, Biometric.GENDER);
//
//			System.out.println("[" + i + "] Starting age biometric prediction...");
//
//			classifier.trainForBiometric(eTrainingSet, Biometric.AGE);
//			resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.AGE);
//
//			System.out.println("[" + i + "] Starting ethnicity biometric prediction...");
//
//			classifier.trainForBiometric(eTrainingSet, Biometric.ETHNICITY);
//			resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.ETHNICITY);
//
//			// Write experiment output to file
//			String featureExtractor = LocalBinaryPatternHandler.getExtractorName(5, 8, 1, false, false, false);
//			String resultSetName = resultSet.getName() + featureExtractor + classifier.getClassifierName();
//			saveResultSet(resultSetName, resultSet);
//
//			System.out.println("[" + i + "] Finished processing iteration");
//		}
	}
}
