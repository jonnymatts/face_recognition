package jonnymatts.facerecognition;

import static java.lang.Math.abs;

import java.util.*;
import java.util.stream.Collectors;

public class PersonDatasetAnalytics {

	private PersonDataset set;
	private HashMap<PersonGender, Integer> genderMap;
	private HashMap<PersonAge, Integer> ageMap;
	private HashMap<PersonAgeFine, Integer> ageFineMap;
	private HashMap<PersonEthnicity, Integer> ethnicityMap;

	PersonDatasetAnalytics(PersonDataset pd) {
		set = new PersonDataset(pd);
		genderMap = new HashMap<PersonGender, Integer>();
		ageMap = new HashMap<PersonAge, Integer>();
		ageFineMap = new HashMap<PersonAgeFine, Integer>();
		ethnicityMap = new HashMap<PersonEthnicity, Integer>();
		updateBiometricMaps();
	}
	
	List<PersonDataset> getOptimizedDatasets(String name, int numberOfGuesses, boolean useAgeFine) {
		double bestFitness = 999;
		List<List<Person>> bestLists = new ArrayList<List<Person>>();
		
		// Calculate the needed sizes for the training and testing lists
		int trainingSetSize = (int)(set.size() * 0.5);
		int testingSetSize = set.size() - trainingSetSize;
		
		for(int i = 0; i < numberOfGuesses; i++) {
			// Create random person lists for the training and testing image set
			List<Person> currentPersonList = new ArrayList<Person>(set.getPersonList());
			List<Person> newTrainingList = new ArrayList<Person>();
			List<Person> newTestingList = new ArrayList<Person>();
			Random rand = new Random();
			
			// Add random people to training list
			for(int j = 0; j < trainingSetSize; j++) {
				Person p = currentPersonList.get(rand.nextInt(currentPersonList.size()));
				newTrainingList.add(p);
				currentPersonList.remove(p);
			}
			
			// Add random people to testing list
			for(int j = 0; j < testingSetSize; j++) {
				Person p = currentPersonList.get(rand.nextInt(currentPersonList.size()));
				newTestingList.add(p);
				currentPersonList.remove(p);
			}
			
			// Calculate fitness current training and testing lists
			double fitness = calculateFitness(newTrainingList, newTestingList, useAgeFine);
			
			// If fitness is lower than best fitness, set best lists to current lists
			if(fitness < bestFitness) {
				bestLists = Arrays.asList(newTrainingList, newTestingList);
				bestFitness = fitness;
			}
		}
		
		// Create new datasets
		String trainingName = name + "_training";
		String testingName = name + "_testing";
		PersonDataset trainingDataset = new PersonDataset(trainingName, bestLists.get(0));
		PersonDataset testingDataset = new PersonDataset(testingName, bestLists.get(1));
		
		return Arrays.asList(trainingDataset, testingDataset);
	}
	
	double calculateFitness(List<Person> trainingList, List<Person> testingList, boolean useAgeFine) {
		
		// Get biometric reports for both training and testing lists
		List<Double> trainingGenderReport = getBiometricReportForList(trainingList, Biometric.GENDER);
		List<Double> trainingAgeReport = useAgeFine ? trainingAgeReport = getBiometricReportForList(trainingList, Biometric.AGEFINE) :
			getBiometricReportForList(trainingList, Biometric.AGE);
		List<Double> trainingEthnicityReport = getBiometricReportForList(trainingList, Biometric.ETHNICITY);
		
		double idealGenderValue = (1d / (PersonGender.values().length - 1));
		double idealAgeValue = useAgeFine ? (1d / (PersonAgeFine.values().length - 1)) : (1d / (PersonAge.values().length - 1));
		double idealEthnicityValue = (1d / (PersonEthnicity.values().length - 1));
		
		double trainingGenderFitness = trainingGenderReport.stream().map(d -> abs(idealGenderValue - d)).reduce(0d, Double::sum);
		double trainingAgeFitness = trainingAgeReport.stream().map(d -> abs(idealAgeValue - d)).reduce(0d, Double::sum);
		double trainingEthnicityFitness = trainingEthnicityReport.stream().map(d -> abs(idealEthnicityValue - d)).reduce(0d, Double::sum);
		
		double totalFitness = trainingGenderFitness + trainingAgeFitness + trainingEthnicityFitness;
		
		return totalFitness;
	}

	void updateBiometricMaps() {
		for (Person p : set.getPersonList()) {

			// Add to genderMap
			if (genderMap.containsKey(p.gender)) {
				genderMap.put(p.gender, (genderMap.get(p.gender) + 1));
			} else {
				genderMap.put(p.gender, 1);
			}

			// Add to ageMap
			if (ageMap.containsKey(p.age)) {
				ageMap.put(p.age, (ageMap.get(p.age) + 1));
			} else {
				ageMap.put(p.age, 1);
			}
			
			// Add to ageFineMap
			if (ageFineMap.containsKey(p.ageFine)) {
				ageFineMap.put(p.ageFine, (ageFineMap.get(p.ageFine) + 1));
			} else {
				ageFineMap.put(p.ageFine, 1);
			}

			// Add to ethnicityMap
			if (ethnicityMap.containsKey(p.ethnicity)) {
				ethnicityMap.put(p.ethnicity, (ethnicityMap.get(p.ethnicity) + 1));
			} else {
				ethnicityMap.put(p.ethnicity, 1);
			}
		}
	}
	
	void addPeopleToPersonList(PersonDataset ds) {
		addPeopleToPersonList(ds.getPersonList());
	}
	

	void addPeopleToPersonList(List<Person> pl) {
		List<Person> currentPersonList = set.getPersonList();
		currentPersonList.addAll(pl);
		currentPersonList.stream().distinct().collect(Collectors.toList());
		set.setPersonList(currentPersonList);
		updateBiometricMaps();
	}
	
	List<Double> getBiometricReportForList(List<Person> personList, Biometric bio) {
		List<Double> biometricList = new ArrayList<Double>();
		
		// Get the amount of people for each section within each biometric 
		switch(bio) {
			case AGE:
				HashMap<PersonAge, Integer> aMap = new HashMap<PersonAge, Integer>();
				for (Person p : set.getPersonList()) {
					// Add to ageMap
					if (aMap.containsKey(p.age)) {
						aMap.put(p.age, (aMap.get(p.age) + 1));
					} else {
						aMap.put(p.age, 1);
					}
				}
				for(PersonAge pa : PersonAge.values()) {
					double valToAdd = aMap.containsKey(pa) ? aMap.get(pa).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
				break;
			case AGEFINE:
				HashMap<PersonAgeFine, Integer> afMap = new HashMap<PersonAgeFine, Integer>();
				for (Person p : set.getPersonList()) {
					// Add to afMap
					if (afMap.containsKey(p.ageFine)) {
						afMap.put(p.ageFine, (afMap.get(p.ageFine) + 1));
					} else {
						afMap.put(p.ageFine, 1);
					}
				}
				for(PersonAgeFine pa : PersonAgeFine.values()) {
					double valToAdd = afMap.containsKey(pa) ? afMap.get(pa).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
				break;
			case ETHNICITY:
				HashMap<PersonEthnicity, Integer> eMap = new HashMap<PersonEthnicity, Integer>();
				for (Person p : set.getPersonList()) {
					// Add to ethnicityMap
					if (eMap.containsKey(p.ethnicity)) {
						eMap.put(p.ethnicity, (eMap.get(p.ethnicity) + 1));
					} else {
						eMap.put(p.ethnicity, 1);
					}
				}
				for(PersonEthnicity pe : PersonEthnicity.values()) {
					double valToAdd = eMap.containsKey(pe) ? eMap.get(pe).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
				break;
			default:
				HashMap<PersonGender, Integer> gMap = new HashMap<PersonGender, Integer>();
				for (Person p : personList) {
					// Add to genderMap
					if (gMap.containsKey(p.gender)) {
						gMap.put(p.gender, (gMap.get(p.gender) + 1));
					} else {
						gMap.put(p.gender, 1);
					}
				}
				for(PersonGender pg : PersonGender.values()) {
					double valToAdd = gMap.containsKey(pg) ? gMap.get(pg).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
		}
		
		// Work out the percentage of each biometric for the total dataset
		int listSize = personList.size();
		List<Double> reportList = biometricList.stream().map(d -> (d / listSize)).collect(Collectors.toList());
		
		return reportList;
	}

	List<Double> getBiometricReportForSet(Biometric bio) {
		List<Double> biometricList = new ArrayList<Double>();
		
		// Get the amount of people for each section within each biometric 
		switch(bio) {
			case AGE:
				for(PersonAge pa : PersonAge.values()) {
					double valToAdd = ageMap.containsKey(pa) ? ageMap.get(pa).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
				break;
			case AGEFINE:
				for(PersonAgeFine pa : PersonAgeFine.values()) {
					double valToAdd = ageFineMap.containsKey(pa) ? ageFineMap.get(pa).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
				break;
			case ETHNICITY:
				for(PersonEthnicity pe : PersonEthnicity.values()) {
					double valToAdd = ethnicityMap.containsKey(pe) ? ethnicityMap.get(pe).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
				break;
			default:
				for(PersonGender pg : PersonGender.values()) {
					double valToAdd = genderMap.containsKey(pg) ? genderMap.get(pg).doubleValue() : 0d;
					biometricList.add(valToAdd);
				}
		}
		
		// Work out the percentage of each biometric for the total dataset
		int listSize = set.size();
		List<Double> reportList = biometricList.stream().map(d -> (d / listSize)).collect(Collectors.toList());
		
		return reportList;
	}

}
