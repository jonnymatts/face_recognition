package jonnymatts.facerecognition;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opencv.core.CvType.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class LocalBinaryPatternHandlerTest {

	LocalBinaryPatternHandler lbph;
	Mat testMat;
	
	@Before
	public void setUp() throws Exception {
		System.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib" );
		testMat = new Mat(3, 3, CvType.CV_64F);
		testMat.put(0, 0, 5);
		testMat.put(0, 1, 2);
		testMat.put(0, 2, 4);
		testMat.put(1, 0, 3);
		testMat.put(1, 1, 4);
		testMat.put(1, 2, 6);
		testMat.put(2, 0, 6);
		testMat.put(2, 1, 7);
		testMat.put(2, 2, 1);
		
		lbph = new LocalBinaryPatternHandler(testMat);
	}
	
	@Test
	public void testLBPHandlerEmptyConstructor() {
		LocalBinaryPatternHandler emptyLbph = new LocalBinaryPatternHandler();
		Mat img = emptyLbph.getImage(); 		
		assertTrue(img.empty());
	}
	
	@Test
	public void testLBPHandlerConstructorWithParam() {
		Mat img = lbph.getImage();
		assertEquals(img, testMat);
	}

	@Test
	public void test3x3LBPCalculation() {
		Double lbp = lbph.calculateLBPForPixel(1, 1, 3);
		assertEquals(lbp, 117, 0.001);
	}
	
	@Test
	public void test3x3LBPCalculationMin() {
		testMat.put(1, 1, 8);
		lbph.setImage(testMat);
		
		Double lbp = lbph.calculateLBPForPixel(1, 1, 3);
		assertEquals(lbp, 0, 0.001);
	}
	
	@Test
	public void test3x3LBPCalculationMax() {
		testMat.put(1, 1, 0);
		lbph.setImage(testMat);
		
		Double lbp = lbph.calculateLBPForPixel(1, 1, 3);
		assertEquals(lbp, 255, 0.001);
	}


}
