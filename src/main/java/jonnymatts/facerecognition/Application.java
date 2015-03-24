package jonnymatts.facerecognition;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import org.opencv.core.*;

import static jonnymatts.facerecognition.ImageHelper.*;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		List<Mat> imgList = Arrays.asList(readImageFromFile("/resources/face_testing_images/1.bmp"),
										  readImageFromFile("/resources/face_testing_images/2.bmp"),
										  readImageFromFile("/resources/face_testing_images/3.bmp"),
										  readImageFromFile("/resources/face_testing_images/4.bmp"),
										  readImageFromFile("/resources/face_testing_images/5.bmp"),
										  readImageFromFile("/resources/face_testing_images/6.bmp"),
										  readImageFromFile("/resources/face_testing_images/7.bmp"),
										  readImageFromFile("/resources/face_testing_images/8.bmp"),
										  readImageFromFile("/resources/face_testing_images/9.bmp"),
										  readImageFromFile("/resources/face_testing_images/10.bmp"),
										  readImageFromFile("/resources/face_testing_images/11.bmp"),
										  readImageFromFile("/resources/face_testing_images/12.bmp"),
										  readImageFromFile("/resources/face_testing_images/13.bmp"),
										  readImageFromFile("/resources/face_testing_images/14.bmp"),
										  readImageFromFile("/resources/face_testing_images/15.bmp"),
										  readImageFromFile("/resources/face_testing_images/16.bmp"),
										  readImageFromFile("/resources/face_testing_images/17.bmp"),
										  readImageFromFile("/resources/face_testing_images/18.bmp"),
										  readImageFromFile("/resources/face_testing_images/19.bmp"),
										  readImageFromFile("/resources/face_testing_images/20.bmp"));
		
		EigenfaceHandler efh = new EigenfaceHandler(imgList);
		
		efh.calculateWeightsForGivenImage(imgList.get(0));
	}
}
