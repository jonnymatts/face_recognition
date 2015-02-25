package jonnymatts.facerecognition;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class ImageHelper {
	
	// Converts 2D matrix to 1D list
	public static List<Double> convertMatToList(Mat input) {
		List<Double> returnList = new ArrayList<Double>();
		
		for(int i = 0; i < input.rows(); i++) {
			for(int j = 0; j < input.cols(); j++) {
				returnList.add((Double)input.get(i,j)[0]);
			}
		}
		return returnList;
	}
	
	public static Mat readImageFromFile(String filename) {
		String dir = System.getProperty("user.dir");
        Mat img = Highgui.imread(dir + filename);
		return img;
	}
	
	// Applies the supplied feature detector to the suppplied image
	public static Mat useFeatureDetector(Mat img, CascadeClassifier cas) {
		Mat grey = new Mat();
        MatOfRect rectMat = new MatOfRect();
        
        // Convert colour image to greyscale 
        Imgproc.cvtColor(img, grey, Imgproc.COLOR_BGR2GRAY);
        
        // Use detector on greyscale image
        cas.detectMultiScale(grey, rectMat);
        
        // For each face detected, draw rectangle onto original image
        for(Rect rect : rectMat.toList()) {
        	Core.rectangle(img, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }
        
		return img;
	}
	
	public static void displayImage(Mat mat) {
		Image img = Mat2BufferedImage(mat);
    	ImageIcon icon=new ImageIcon(img);
    	JFrame frame=new JFrame();
    	frame.setLayout(new FlowLayout());        
    	frame.setSize(img.getWidth(null)+50, img.getHeight(null)+50);     
    	JLabel lbl=new JLabel();
    	lbl.setIcon(icon);
    	frame.add(lbl);
    	frame.setVisible(true);
    	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

	public static BufferedImage Mat2BufferedImage(Mat m) {
		int type = BufferedImage.TYPE_BYTE_GRAY;
		if ( m.channels() > 1 ) {
			type = BufferedImage.TYPE_3BYTE_BGR;
		}
		int bufferSize = m.channels()*m.cols()*m.rows();
		byte [] b = new byte[bufferSize];
		m.get(0,0,b); // get all the pixels
		BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);  
		return image;
	}
	
}
