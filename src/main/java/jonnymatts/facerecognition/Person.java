package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.*;

import org.opencv.core.Mat;

public class Person {
	public PersonGender gender;
	public PersonAge age;
	public PersonEthnicity ethnicity;
	public Mat colourImage;
	public Mat depthImage;
	
	Person(int g, int a, int e, String colourImagePath, String depthImagePath) {
		gender = PersonGender.valueOf(g);
		age = PersonAge.valueOf(a);
		ethnicity = PersonEthnicity.valueOf(e);
		colourImage = readImageFromFile(colourImagePath);
		depthImage = readImageFromFile(depthImagePath);
	}
}
