package jonnymatts.facerecognition;

import static org.junit.Assert.*;
import static java.lang.Math.ceil;
import static java.lang.Math.floor;
import static jonnymatts.facerecognition.ImageHelper.*;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.objdetect.CascadeClassifier;

public class ImageHelperTest {

	String dir;
	CascadeClassifier cas;

	@Before
	public void setUp() throws Exception {
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		dir = System.getProperty("user.dir");
	}

	@Test
	public void testImageReadCorrectInputJpg() {
		Mat img = readImageFromFile("/resources/villa.jpg");

		assertFalse(img.empty());
		assertEquals(img.height(), 764);
		assertEquals(img.channels(), 3);
	}

	@Test
	public void testImageReadCorrectInputBmp() {
		Mat img = readImageFromFile("/resources/lena.bmp");

		assertFalse(img.empty());
		assertEquals(img.height(), 512);
		assertEquals(img.channels(), 3);
	}

	@Test
	public void testImageReadError() {
		Mat img = readImageFromFile("/resources/non-existent.jpg");

		assertTrue(img.empty());
		assertEquals(img.height(), 0);
	}

	@Test
	public void testBilinearInterpolation() {
		Mat testMat = new Mat(3, 3, CvType.CV_64F);
		testMat.put(0, 0, 1);
		testMat.put(0, 1, 2);
		testMat.put(0, 2, 3);
		testMat.put(1, 0, 4);
		testMat.put(1, 1, 5);
		testMat.put(1, 2, 6);
		testMat.put(2, 0, 7);
		testMat.put(2, 1, 8);
		testMat.put(2, 2, 9);
		
		// Find values of the 4 surrounding cells
		double px = 1.3;
		double py = 0.5;
		
		double xMin = floor(px);
		int xMinIndex = (int) xMin;
		double xMax = ceil(px);
		int xMaxIndex = (int) xMax;
		double yMin = floor(py);
		int yMinIndex = (int) yMin;
		double yMax = ceil(py);
		int yMaxIndex = (int) yMax;
		double x1 = testMat.get(yMinIndex, xMinIndex)[0];
		double x2 = testMat.get(yMinIndex, xMaxIndex)[0];
		double x3 = testMat.get(yMaxIndex, xMinIndex)[0];
		double x4 = testMat.get(yMaxIndex, xMaxIndex)[0];

		double intVal = bilinearInterpolation(px, py, xMin, xMax, yMin, yMax, x1, x2, x3, x4);

		assertEquals(intVal, 3.8, 0.001);
	}

}
