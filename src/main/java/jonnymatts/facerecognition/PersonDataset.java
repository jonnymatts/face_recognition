package jonnymatts.facerecognition;

import java.util.List;
import java.util.stream.Collectors;

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
	
	List<String> getListOfUnProcessedPeople() {
		return personList.stream().filter(p -> !p.getIsPreprocessed()).map(p -> p.name).collect(Collectors.toList());
	}
}
