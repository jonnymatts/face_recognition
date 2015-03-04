package jonnymatts.facerecognition;

import org.opencv.core.*;

import static jonnymatts.facerecognition.ImageHelper.*;

public class Application {
	public static void main(String[] args) {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");

		Mat img = readImageFromFile("/resources/youngme.jpg");

		LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler(8, 1, false, false, true);

		Mat redImg = lbph.calculateLBP(img, LBPColour.RED);
		Mat greenImg = lbph.calculateLBP(img, LBPColour.GREEN);
		Mat blueImg = lbph.calculateLBP(img, LBPColour.BLUE);

		System.out.println(lbph.findFeatureVector(img, 5));
		
		displayImage(redImg);
		displayImage(greenImg);
		displayImage(blueImg);
	}
}
