package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		// Read in result set
		PersonDataset results = readResultSet("/resources/results/EURECOM_1_testing_RISE_25_SVM_.txt");
		
		// Check personList of resultSet to see if prediction was correct for each biometric
		List<Boolean> genderBoolList = results.checkPredictedClassesForBiometric(Biometric.GENDER);
		List<Boolean> ageBoolList = results.checkPredictedClassesForBiometric(Biometric.AGE);
		List<Boolean> ethnicityBoolList = results.checkPredictedClassesForBiometric(Biometric.ETHNICITY);
		
		// Convert boolLists into a percentage
		double genderCorrect = findPercentageCorrect(genderBoolList);
		double ageCorrect = findPercentageCorrect(ageBoolList);
		double ethnicityCorrect = findPercentageCorrect(ethnicityBoolList);
		
		// Print out results
		System.out.println("genderCorrect: " + genderCorrect);
		System.out.println("ageCorrect: " + ageCorrect);
		System.out.println("ethnicityCorrect: " + ethnicityCorrect);
		
	}
}
