package jonnymatts.facerecognition;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.*;

import static jonnymatts.facerecognition.ImageHelper.*;

public class Application 
{	
    public static void main( String[] args )
    {   
    	// Load .dylib file for openCV
    	System.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib" );
    	
    	// Define CascadeClassifier for Viola-Jones face detection
    	CascadeClassifier cas = new CascadeClassifier("/Users/jonnymatts/dev/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml");
        
        Mat img = readImageFromFile("/resources/lena.bmp");
        
        // Detect faces in image using Viola-Jones
        Mat faceDetectedImg = useFeatureDetector(img, cas);
        
        LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler();
        
        // Calculate the LBP of the image
        System.out.println(lbph.findFeatureVector(img, 3, 8, 1));
        
    }
}
