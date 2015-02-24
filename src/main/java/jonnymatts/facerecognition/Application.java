package jonnymatts.facerecognition;

import org.opencv.core.*;
import org.opencv.highgui.*;
import org.opencv.imgproc.*;
import org.opencv.objdetect.*;

import static jonnymatts.facerecognition.ImageHelper.*;

public class Application 
{	
    public static void main( String[] args )
    {   
    	System.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib" );
    	CascadeClassifier cas = new CascadeClassifier("/Users/jonnymatts/dev/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml");
        
    	String dir = System.getProperty("user.dir");
        String youngImagePath = dir + "/resources/youngme.jpg";
        
        Mat youngImg = Highgui.imread(youngImagePath);
        
        Mat grey = new Mat();
        
        MatOfRect rectMat = new MatOfRect();
        
        Imgproc.cvtColor(youngImg, grey, Imgproc.COLOR_BGR2GRAY);
        
        cas.detectMultiScale(grey, rectMat);
        
        for(Rect rect : rectMat.toList()) {
        	Core.rectangle(youngImg, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }
        
        double[] vals = youngImg.get(435, 435);
        
        displayImage(Mat2BufferedImage(youngImg));
    }
}
