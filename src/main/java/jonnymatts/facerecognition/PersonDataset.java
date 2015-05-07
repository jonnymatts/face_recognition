package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.findPercentageCorrect;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class PersonDataset {
	private String name;
	private List<Person> personList;
	
	public void setName(String n) {
		name = n;
	}
	
	public String getName() {
		return name;
	}
	
	public void setPersonList(List<Person> pl) {
		personList = pl;
	}
	
	public List<Person> getPersonList() {
		return personList;
	}
	
	public int size() {
		return personList.size();
	}
	
	PersonDataset(String n, List<Person> pl) {
		name = n;
		personList = pl;
	}
	
	PersonDataset(PersonDataset pd) {
		name = pd.name;
		personList = copyPersonList(pd.getPersonList());
	}
	
	List<Person> copyPersonList(List<Person> pl) {
		List<Person> returnList = new ArrayList<Person>();
		for(Person p : pl) {
			returnList.add(new Person(p));
		}
		return returnList;
	}
	
	List<Person> resizeImages(int dimension) {
		List<Person> pList = personList;
		for(Person p : pList) {
			Imgproc.resize(p.colourImage, p.colourImage, new Size(dimension, dimension));
			Imgproc.resize(p.depthImage, p.depthImage, new Size(dimension, dimension));
		}
		return pList;
	}
	
	double getLargestValueInDataset() {
		return personList.stream().map(p -> p.getFeatureVector().stream().sorted(Collections.reverseOrder()).collect(Collectors.toList()).get(0)).sorted(Collections.reverseOrder()).collect(Collectors.toList()).get(0);
	}
	
	DatasetResult getDatasetResult(long time, long train, long predict, boolean useAgeFine) {
		DatasetResult dr = getDatasetResult(useAgeFine);
		dr.extractorTime = time;
		dr.trainingTime = train;
		dr.predictingTime = predict;
		return dr;
	}
	
	DatasetResult getDatasetResult(boolean useAgeFine) {
		double genderCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.GENDER));
		double ageCorrect;
		if(useAgeFine){
			ageCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.AGEFINE));
		} else {
			ageCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.AGE));
		}
		double ethnicityCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.ETHNICITY));
		return new DatasetResult(name, genderCorrect, ageCorrect, ethnicityCorrect);
	}
	
	List<Boolean> checkPredictedClassesForBiometric(Biometric biometric) {
		switch(biometric) {
			case AGE:
				return personList.stream().map(p -> p.age == p.predictedAge).collect(Collectors.toList());
			case AGEFINE:
				return personList.stream().map(p -> p.ageFine == p.predictedAgeFine).collect(Collectors.toList());
			case ETHNICITY:
				return personList.stream().map(p -> p.ethnicity == p.predictedEthnicity).collect(Collectors.toList());
			default:
				return personList.stream().map(p -> p.gender == p.predictedGender).collect(Collectors.toList());
		}
	}
	
	Dataset createKNNDatasetForBiometric(Biometric biometric) {
		Dataset data = new DefaultDataset();
		List<Instance> instanceList = personList.stream().map(p -> p.createKNNInstanceForBiometric(biometric)).collect(Collectors.toList());
		for(Instance i : instanceList) {
			data.add(i);
		}
		return data;
	}
	
	List<Mat> getDepthImageList() {
		return personList.stream().map(i -> i.depthImage).collect(Collectors.toList());
	}
	
	List<Mat> getColourImageList() {
		return personList.stream().map(i -> i.colourImage).collect(Collectors.toList());
	}
	
	List<Person> getListOfUnProcessedPeople() {
		return personList.stream().filter(p -> !p.getIsPreprocessed()).collect(Collectors.toList());
	}
}
