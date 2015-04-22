package jonnymatts.facerecognition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.io.IOException;

import net.sf.javaml.core.Dataset;

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
		
		
		PersonDataset set1 = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting1.txt");
		PersonDataset set2 = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting2.txt");
//		set.setPersonList(performPreprocessing(set, 256));
//		PersonDataset trainingSet = performGLBPFeatureExtractionOnDataset(set, 8, 1);
//		set = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting2.txt");
//		set.setPersonList(performPreprocessing(set, 256));
//		PersonDataset testingSet = performGLBPFeatureExtractionOnDataset(set, 8, 1);
		
//		PersonDataset set = readDataset("/resources/datasets/test_dataset_training.txt");
		set1.setPersonList(performPreprocessing(set1, 256));
		set2.setPersonList(performPreprocessing(set2, 256));
		
		PersonDatasetAnalytics pda = new PersonDatasetAnalytics(set1);
		pda.addPeopleToPersonList(set2.getPersonList());
		List<PersonDataset> datasets = pda.getOptimizedDatasets("EURECOM", 100);
		PersonDataset trainingSet = datasets.get(0);
		PersonDataset testingSet = datasets.get(1);
		
		trainingSet = performGLBPFeatureExtractionOnDataset(trainingSet, 8, 1);
		testingSet = performGLBPFeatureExtractionOnDataset(testingSet, 8, 1);
		
		KNNHandler knn = new KNNHandler();
		knn.trainForBiometric(trainingSet, Biometric.GENDER, 3);
		testingSet = knn.predictClassesForBiometric(testingSet, Biometric.GENDER);
		System.out.println(findPercentageCorrect(testingSet.checkPredictedClassesForBiometric(Biometric.GENDER)));
		
		
//		KNNHandler knn = new KNNHandler("/resources/classifier_inputs/knn/test_dataset_training_age_RISE_2015_04_10_15_49_40.data", 1);
//		List<Boolean> boolList = knn.predictClassOfTestData("/resources/classifier_inputs/knn/test_dataset_testing2_age_RISE_2015_04_10_15_49_59.data");
//		System.out.println(boolList);
		
//		SVMHandler svmh = new SVMHandler();
//		svmh.trainSVMForBiometric(trainingSet, Biometric.AGE);
//		testingSet = svmh.predictClasses(testingSet, Biometric.AGE);
//		System.out.println(findPercentageCorrect(testingSet.checkPredictedClassesForBiometric(Biometric.AGE)));
	}
}
