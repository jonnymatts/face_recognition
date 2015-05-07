package jonnymatts.facerecognition;

public interface ProjectClassifier {

	void trainForBiometric(PersonDataset eTrainingSet, Biometric gender);

	PersonDataset predictClassesForBiometric(PersonDataset eTestingSet,
			Biometric gender);

	String getClassifierName();

}
