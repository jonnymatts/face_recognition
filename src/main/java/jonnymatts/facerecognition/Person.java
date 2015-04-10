package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;

public class Person {
	public PersonGender gender;
	public PersonAge age;
	public PersonEthnicity ethnicity;
	public Mat colourImage;
	public Mat depthImage;
	private List<Double> featureVector;
	
	public void setFeatureVector(List<Double> fv) {
		featureVector = fv;
	}
	
	public List<Double> getFeatureVector() {
		return featureVector;
	}
	
	Person(int g, int a, int e, String colourImagePath, String depthImagePath) {
		gender = PersonGender.valueOf(g);
		age = PersonAge.valueOf(a);
		ethnicity = PersonEthnicity.valueOf(e);
		colourImage = readImageFromFile(colourImagePath);
		depthImage = readImageFromFile(depthImagePath);
		featureVector = new ArrayList<Double>();
	}
}