package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		List<Integer> intList = Arrays.asList(1,2,3,4,5);
		
		for(Integer i : intList){
			// Load the previously generated training and testing datasets
			PersonDataset trainingSet = readDataset("/resources/datasets/EURECOM_" + i + "_training.txt");
			PersonDataset testingSet = readDataset("/resources/datasets/EURECOM_" + i + "_testing.txt");
			
			System.out.println("[" + i + "] Starting image pre-processing...");
			
			// Pre-process all the images
			trainingSet.setPersonList(performPreprocessing(trainingSet, 256));
			testingSet.setPersonList(performPreprocessing(testingSet, 256));
			
			// Perform feature extraction on both datasets
			
			System.out.println("[" + i + "] Starting feature extraction on training set...");
			
			int pixels = 25;
			
			PersonDataset eTrainingSet = performRISEFeatureExtractionOnDataset(trainingSet, pixels);
			
			System.out.println("[" + i + "] Starting feature extraction on testing set...");
			
			PersonDataset eTestingSet = performRISEFeatureExtractionOnDataset(testingSet, pixels);
			
			System.out.println("[" + i + "] Starting gender biometric prediction...");
			
			// Train the classifer, then predict the class of testing data for each biometric
			SVMHandler classifier = new SVMHandler();
			classifier.trainForBiometric(eTrainingSet, Biometric.GENDER);
			PersonDataset resultSet = classifier.predictClassesForBiometric(eTestingSet, Biometric.GENDER);
			
			System.out.println("[" + i + "] Starting age biometric prediction...");
			
			classifier.trainForBiometric(eTrainingSet, Biometric.AGE);
			resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.AGE);
			
			System.out.println("[" + i + "] Starting ethnicity biometric prediction...");
			
			classifier.trainForBiometric(eTrainingSet, Biometric.ETHNICITY);
			resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.ETHNICITY);
			
			// Write experiment output to file
			String featureExtractor = RISEHandler.getExtractorName(pixels);
			String resultSetName = resultSet.getName() + featureExtractor + classifier.getClassifierName(); 
			saveResultSet(resultSetName, resultSet);
			
			System.out.println("[" + i + "] Finished processing iteration");
		}
	}
}
