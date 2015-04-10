package jonnymatts.facerecognition;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileStore;
import java.nio.file.FileSystems;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import com.google.common.collect.Lists;

import static jonnymatts.facerecognition.ImageHelper.*;
import static jonnymatts.facerecognition.ApplicationUtil.*;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		KNNHandler knn = new KNNHandler("/resources/classifier_inputs/knn/test_dataset_training_RISE_2015_04_10_14_41_09.data", 1);
		List<Boolean> boolList = knn.predictClassOfTestData("/resources/classifier_inputs/knn/test_dataset_testing2_RISE_2015_04_10_15_01_13.data");
		System.out.println(boolList);
	}
}
