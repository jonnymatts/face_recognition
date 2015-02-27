package jonnymatts.facerecognition;

import org.opencv.core.*;

import static jonnymatts.facerecognition.ImageHelper.*;

public class Application {
	public static void main(String[] args) {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");

		Mat img = readImageFromFile("/resources/lena.bmp");

		LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler(8, 1,
				true);

		Mat newImg = lbph.calculateLBP(img);

		displayImage(newImg);
	}
}
