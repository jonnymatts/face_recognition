package jonnymatts.facerecognition;

import java.util.List;
import java.io.IOException;

import static jonnymatts.facerecognition.ImageHelper.*;
import static jonnymatts.facerecognition.ApplicationUtil.*;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
//		PersonDataset set = readDataset("/resources/datasets/test_dataset_training.txt");
//		set = performRISEFeatureExtractionOnDataset(set, 25);
//		writeResultSetToFilesForKNNClassifier(set, "RISE");
//		
//		set = readDataset("/resources/datasets/test_dataset_testing.txt");
//		set = performRISEFeatureExtractionOnDataset(set, 25);
//		writeResultSetToFilesForKNNClassifier(set, "RISE");
//		
//		set = readDataset("/resources/datasets/test_dataset_testing2.txt");
//		set = performRISEFeatureExtractionOnDataset(set, 25);
//		writeResultSetToFilesForKNNClassifier(set, "RISE");
		
		KNNHandler knn = new KNNHandler("/resources/classifier_inputs/knn/test_dataset_training_age_RISE_2015_04_10_15_49_40.data", 1);
		List<Boolean> boolList = knn.predictClassOfTestData("/resources/classifier_inputs/knn/test_dataset_testing2_age_RISE_2015_04_10_15_49_59.data");
		System.out.println(boolList);
	}
}
