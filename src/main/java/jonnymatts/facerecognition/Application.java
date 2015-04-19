package jonnymatts.facerecognition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.io.IOException;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import com.google.common.primitives.Doubles;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import static jonnymatts.facerecognition.ImageHelper.*;
import static jonnymatts.facerecognition.ApplicationUtil.*;
import static java.lang.Math.*;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		PersonDataset set = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting1.txt");
//		set.setPersonList(performPreprocessing(set, 256));
//		System.out.println(set.getListOfUnProcessedPeople().size());	
		
//		set = performRISEFeatureExtractionOnDataset(set, 25);
		
//		
//		set = readDataset("/resources/datasets/test_dataset_testing.txt");
//		set = performRISEFeatureExtractionOnDataset(set, 25);
//		writeResultSetToFilesForKNNClassifier(set, "RISE");
//		
//		PersonDataset set = readDataset("/resources/datasets/test_dataset_testing.txt");
//		PersonDataset trainingSet = performGLBPFeatureExtractionOnDataset(set, 8, 1);
//		set = readDataset("/resources/datasets/test_dataset_testing2.txt");
//		PersonDataset testingSet = performGLBPFeatureExtractionOnDataset(set, 8, 1);
		
		
//		writeResultSetToFileForSVMClassifier(set, "RISE", Biometric.GENDER);
		
//		KNNHandler knn = new KNNHandler("/resources/classifier_inputs/knn/test_dataset_training_age_RISE_2015_04_10_15_49_40.data", 1);
//		List<Boolean> boolList = knn.predictClassOfTestData("/resources/classifier_inputs/knn/test_dataset_testing2_age_RISE_2015_04_10_15_49_59.data");
//		System.out.println(boolList);
		
//		SVMHandler svmh = new SVMHandler();
//		svmh.trainSVMForBiometric(trainingSet, Biometric.ETHNICITY);
//		System.out.println(svmh.predictClasses(testingSet, Biometric.ETHNICITY));
	}
}
