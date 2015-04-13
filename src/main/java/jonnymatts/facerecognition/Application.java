package jonnymatts.facerecognition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.io.IOException;

import org.opencv.core.Mat;

import com.google.common.primitives.Doubles;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import static jonnymatts.facerecognition.ImageHelper.*;
import static jonnymatts.facerecognition.ApplicationUtil.*;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		GradientLBPHandler gh = new GradientLBPHandler(8, 1);
		
		Mat image = readImageFromFile("/resources/face_testing_images/2.bmp");
		Mat depthImage = readImageFromFile("/resources/face_testing_images/2_depth.bmp");
		
		System.out.println(gh.findFeatureVector(image, depthImage));
	}
}
