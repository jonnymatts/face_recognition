package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.readImageFromFile;

import java.util.ArrayList;
import java.util.List;

import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

import org.opencv.core.Mat;

import com.google.common.primitives.Doubles;

public class Person {
	public String name;
	public PersonGender gender;
	public PersonGender predictedGender;
	public PersonAge age;
	public PersonAge predictedAge;
	public PersonEthnicity ethnicity;
	public PersonEthnicity predictedEthnicity;
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
	
	Person(String n, String g, String pg, String a, String pa, String e, String pe) {
		name = n;
		gender = PersonGender.valueOf(g);
		predictedGender = PersonGender.valueOf(pg);
		age = PersonAge.valueOf(a);
		predictedAge = PersonAge.valueOf(pa);
		ethnicity = PersonEthnicity.valueOf(e);
		predictedEthnicity = PersonEthnicity.valueOf(pe);
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
	
	Instance createKNNInstanceForBiometric(Biometric biometric) {
		double[] valArray = Doubles.toArray(featureVector);
		switch(biometric) {
			case AGE: return new DenseInstance(valArray, age.getValue());
			case ETHNICITY: return new DenseInstance(valArray, ethnicity.getValue());
			default: return new DenseInstance(valArray, gender.getValue());
		}
	}
}
