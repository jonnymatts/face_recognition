package jonnymatts.facerecognition;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class ImageHelper {
	
	public static List<LBPColour> lbpColourList = Arrays.asList(LBPColour.RED, LBPColour.GREEN, LBPColour.BLUE);
	
	public static boolean compareImages(Mat im1, Mat im2) {
		Mat op = new Mat();
		Core.compare(im1, im2, op, Core.CMP_NE);
		return (Core.countNonZero(op) == 0);
	}
	
	public static Mat convertRGBtoYUV(Mat in) {
		Mat yuvImage = new Mat(in.rows(), in.cols(), CvType.CV_32FC3);
		
		for (int i = 0; i < in.cols(); i++) {
			for (int j = 0; j < in.rows(); j++) {
				double[] bgrArray = in.get(j,i);
				double bVal = bgrArray[0];
				double gVal = bgrArray[1];
				double rVal = bgrArray[2];
				float yVal = (float)((0.29f * rVal) + (0.587f * gVal) + (0.114f * bVal));
				float uVal = (float)(0.492f * (bVal - yVal));
				float vVal = (float)(0.877f * (rVal - yVal));
				float[] yuvArray = {yVal, uVal, vVal};
				yuvImage.put(j, i, yuvArray);
			}
		}
		
		return yuvImage;
	}

	public static double bilinearInterpolation(double x, double y, double xMin, double xMax, double yMin, double yMax, double x1, double x2, double x3, double x4) {

		double xDistToMax = (xMax - x) / (xMax - xMin);
		double xDistToMin = (x - xMin) / (xMax - xMin);
		double yDistToMax = (yMax - y) / (yMax - yMin);
		double yDistToMin = (y - yMin) / (yMax - yMin);

		double returnVal;

		// Y-axis interpolation
		if (x % 1 == 0) {
			returnVal = (yDistToMax * x1) + (yDistToMin * x3);
		}
		// X-axis interpolation
		else if (y % 1 == 0) {
			returnVal = (xDistToMax * x1) + (xDistToMin * x2);
		}
		// X and Y-axis interpolation
		else {

			// Find two interpolated points down the x-axis
			double xInt1 = (xDistToMax * x1) + (xDistToMin * x2);
			double xInt2 = (xDistToMax * x3) + (xDistToMin * x4);

			// Interpolate between the two points
			double yInt = (yDistToMax * xInt1) + (yDistToMin * xInt2);

			returnVal = yInt;
		}
		return returnVal;
	}

	// Converts 2D matrix to 1D list
	public static List<Double> convertMatToList(Mat input, int index) {
		List<Double> returnList = new ArrayList<Double>();

		for (int i = 0; i < input.rows(); i++) {
			for (int j = 0; j < input.cols(); j++) {
				returnList.add((Double) input.get(i, j)[index]);
			}
		}
		return returnList;
	}

	public static Mat readImageFromFile(String filename) {
		String dir = System.getProperty("user.dir");
		Mat img = Highgui.imread(dir + filename);
		return img;
	}
	
	public static File loadFile(String filename) {
		String dir = System.getProperty("user.dir");
		return new File(dir + filename);
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
		for (Rect rect : rectMat.toList()) {
			Core.rectangle(img, new Point(rect.x, rect.y), new Point(rect.x
					+ rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
		}

		return img;
	}

	public static void displayImage(Mat mat) {
		Image img = Mat2BufferedImage(mat);
		ImageIcon icon = new ImageIcon(img);
		JFrame frame = new JFrame();
		frame.setLayout(new FlowLayout());
		frame.setSize(img.getWidth(null) + 50, img.getHeight(null) + 50);
		JLabel lbl = new JLabel();
		lbl.setIcon(icon);
		frame.add(lbl);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static BufferedImage Mat2BufferedImage(Mat m) {
		m.convertTo(m, CvType.CV_8U);
		int type = BufferedImage.TYPE_BYTE_GRAY;
		if (m.channels() > 1) {
			type = BufferedImage.TYPE_3BYTE_BGR;
		}
		int bufferSize = m.channels() * m.cols() * m.rows();
		byte[] b = new byte[bufferSize];
		m.get(0, 0, b); // get all the pixels
		BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster()
				.getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);
		return image;
	}

}
