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
        
        Mat img = readImageFromFile("/resources/lena.bmp");
        
        LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler();
        
        Mat newImg = lbph.calculateLBP(img, 8, 5);       
        
        displayImage(newImg);
    }
}
