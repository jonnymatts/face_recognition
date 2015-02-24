package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.*;

import org.opencv.core.Mat;

public class LocalBinaryPatternHandler {
	
	private Mat img;
	
	public void setImage(Mat input) {
		img = input;
	}
	
	public Mat getImage() {
		return img;
	}
	
	public LocalBinaryPatternHandler() {
		img = new Mat();
	}
	
	public LocalBinaryPatternHandler(Mat input) {
		img = input;
	}
	
	public List<List<Double>> findFeatureVector() {
		return new ArrayList< List<Double>>();
	}
	
	Double calculateLBPForPixel(int x, int y, int size) {
		
		// Find radius of neighbourhood
		int nbhRadius = ((size + 1) / 2) - 1;
		
		// Find neighbourhood for central pixel
		Mat nbh = img.submat(x-nbhRadius, x+nbhRadius+1, y-nbhRadius, y+nbhRadius+1);
		
		// Convert nbh to 1D list
		List<Double> nbhList = convertMatToList(nbh);
		
		// Remove and preserve central pixel value from list
		int centrePixelIndex = (size * size) / 2;
		Double centrePixelValue = nbhList.get(centrePixelIndex);
		nbhList.remove(centrePixelIndex);
		
		// Convert list value to either 1 or 0
		// if(value < centrePixelValue) 0 else 1
		List<Integer> nbhBinList = nbhList.stream().map(d -> (d < centrePixelValue)).map(b -> b ? 0 : 1).collect(Collectors.toList());;
		
		// Find the sum of values multiplied by the corresponding power of 2
		Double sum = 0d;
		for(int i = 0; i < nbhList.size(); i++){
			sum = sum + nbhBinList.get(i) * (Math.pow(2, i));
		}
		
		return sum;
	}

}
