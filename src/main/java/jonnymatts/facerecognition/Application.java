package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.*;

import java.io.IOException;
import java.util.List;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		// Load both sittings from the EURECOM dataset
		PersonDataset set1 = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting1.txt");
		PersonDataset set2 = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting2.txt");
		
		// Pre-process all the images
		set1.setPersonList(performPreprocessing(set1, 256));
		set2.setPersonList(performPreprocessing(set2, 256));
		
		// Create optimized training and testing sets using genetic algorithm
		PersonDatasetAnalytics pda = new PersonDatasetAnalytics(set1);
		pda.addPeopleToPersonList(set2);
		List<PersonDataset> datasets = pda.getOptimizedDatasets("EURECOM_initial", 100);
		PersonDataset trainingSet = datasets.get(0);
		PersonDataset testingSet = datasets.get(1);
		
		// Save datasets for further use
		saveDataset(trainingSet);
		saveDataset(testingSet);
	}
}
