package jonnymatts.facerecognition;

import java.util.List;

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
	
	PersonDataset(String n, List<Person> pl) {
		name = n;
		personList = pl;
	}
}
