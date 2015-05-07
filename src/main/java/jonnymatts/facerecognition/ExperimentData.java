package jonnymatts.facerecognition;

public class ExperimentData {

	String name;
	PersonDataset trainingSet;
	PersonDataset testingSet;
	boolean useAgeFine;

	ExperimentData(String n, PersonDataset train, PersonDataset test, boolean a) {
		name = n;
		trainingSet = train;
		testingSet = test;
		useAgeFine = a;
	}
}
