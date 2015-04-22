package jonnymatts.facerecognition;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class LocalBinaryPatternHandlerTest {

	LocalBinaryPatternHandler lbph;
	Mat testMat;

	@Before
	public void setUp() throws Exception {
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
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

		lbph = new LocalBinaryPatternHandler(8, 1, false, false, false);
	}

	@Test
	public void testLBPCalculation() {
		Double lbp = lbph.calculateLBPForPixel(testMat, 1, 1, LBPColour.GREY);
		assertEquals(lbp, 176, 0.001);
	}

	@Test
	public void testLBPCalculationMin() {
		testMat.put(1, 1, 8);

		Double lbp = lbph.calculateLBPForPixel(testMat, 1, 1, LBPColour.GREY);
		assertEquals(lbp, 0, 0.001);
	}

	@Test
	public void testLBPCalculationMax() {
		testMat.put(1, 1, 0);

		Double lbp = lbph.calculateLBPForPixel(testMat, 1, 1, LBPColour.GREY);
		assertEquals(lbp, 255, 0.001);
	}

	@Test
	public void testIsUniformPattern() {
		assertEquals(lbph.findFeatureVector(testMat, 5).get(0).size(), 256);

		lbph.setUseUniformPatterns(true);
		assertEquals(lbph.findFeatureVector(testMat, 5).get(0).size(), 59);
	}
	
	@Test
	public void testIntegerToListConversions() {
		List<Integer> l = lbph.convertIntegerToBinaryList(13);
		int answer = lbph.convertBinaryListToInteger(l);
		assertEquals(answer, 13);
	}
	
	@Test
	public void testRotationInvariance() { 
		List<Integer> l = Arrays.asList(1,0,1,1);
		List<Integer> r = lbph.findRotationInvariantSequence(l);
		assertEquals(r, Arrays.asList(1,1,1,0));
	}
	
	@Test
	public void testIsRotationInvariant() {
		lbph.setUseRotationInvariance(true);
		lbph.setUseUniformPatterns(false);
		assertEquals(lbph.findFeatureVector(testMat, 5).get(0).size(), 37);
	}
	
	@Test
	public void testIsRotationInvariantAndUsesUniformPatterns () {
		lbph.setUseRotationInvariance(true);
		lbph.setUseUniformPatterns(true);
		assertEquals(lbph.findFeatureVector(testMat, 5).get(0).size(), 9);
	}
	
	@Test
	public void testUsesRGB () {
		lbph.setUseRotationInvariance(false);
		lbph.setUseUniformPatterns(false);
		lbph.setUseRGB(true);
		assertEquals(lbph.findFeatureVector(testMat, 5).size(), (75));
	}
}
