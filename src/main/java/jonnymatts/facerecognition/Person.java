package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;

public class Person {
	public String name;
	public PersonGender gender;
	public PersonAge age;
	public PersonEthnicity ethnicity;
	public String colourImagePath;
	public String depthImagePath;
	public Mat colourImage;
	public Mat depthImage;
	private boolean isPreprocessed;
	private List<Double> featureVector;
	
	public boolean getIsPreprocessed() {
		return isPreprocessed;
	}

	public void setIsPreprocessed(boolean i) {
		isPreprocessed = i;
	}

	public void setFeatureVector(List<Double> fv) {
		featureVector = fv;
	}
	
	public List<Double> getFeatureVector() {
		return featureVector;
	}
	
	Person(String n, int g, int a, int e, String cPath, String dPath) {
		name = n;
		gender = PersonGender.valueOf(g);
		age = PersonAge.valueOf(a);
		ethnicity = PersonEthnicity.valueOf(e);
		colourImagePath = cPath;
		depthImagePath = dPath;
		colourImage = readImageFromFile(colourImagePath);
		depthImage = readImageFromFile(depthImagePath);
		isPreprocessed = false;
		featureVector = new ArrayList<Double>();
	}
}
