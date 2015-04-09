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
		
//		Mat img = readImageFromFile("/resources/face_testing_images/1.bmp");
//		Mat depthImg = normaliseDepthImage(readImageFromFile("/resources/face_testing_images/1_depth.bmp"));
//		
//		RISEHandler rh = new RISEHandler(3, 15, 1, (img.rows() / 10), 0.02, 15, 230, 30, 25);
//		
//		System.out.println(rh.findFeatureVector(img, depthImg));
		
		List<Person> pList = readDataset("/resources/datasets/test_dataset.txt");
		
		Person p = pList.get(0);
		displayImage(p.colourImage);
		displayImage(p.depthImage);
	}
}
