package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.findPercentageCorrect;

import java.util.List;
import java.util.stream.Collectors;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;

import org.opencv.core.Mat;

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
	
	DatasetResult getDatasetResult() {
		double genderCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.GENDER));
		double ageCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.AGE));
		double ethnicityCorrect = findPercentageCorrect(checkPredictedClassesForBiometric(Biometric.ETHNICITY));
		return new DatasetResult(genderCorrect, ageCorrect, ethnicityCorrect);
	}
	
	List<Boolean> checkPredictedClassesForBiometric(Biometric biometric) {
		switch(biometric) {
			case AGE:
				return personList.stream().map(p -> p.age == p.predictedAge).collect(Collectors.toList());
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
