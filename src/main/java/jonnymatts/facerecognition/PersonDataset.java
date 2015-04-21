package jonnymatts.facerecognition;

import java.util.List;
import java.util.stream.Collectors;

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
	
	List<Mat> getDepthImageList() {
		return personList.stream().map(i -> i.depthImage).collect(Collectors.toList());
	}
	
	List<Mat> getColourImageList() {
		return personList.stream().map(i -> i.colourImage).collect(Collectors.toList());
	}
	
	List<String> getListOfUnProcessedPeople() {
		return personList.stream().filter(p -> !p.getIsPreprocessed()).map(p -> p.name).collect(Collectors.toList());
	}
}
