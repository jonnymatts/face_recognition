package jonnymatts.facerecognition.test;

import static org.junit.Assert.*;
import static jonnymatts.facerecognition.ImageHelper.*;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.Mat;
import org.opencv.objdetect.CascadeClassifier;

public class ImageHelperTest {

	String dir;
	CascadeClassifier cas;
	
	@Before
	public void setUp() throws Exception {
		System.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib" );
    	CascadeClassifier cas = new CascadeClassifier("/Users/jonnymatts/dev/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml");
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

}
