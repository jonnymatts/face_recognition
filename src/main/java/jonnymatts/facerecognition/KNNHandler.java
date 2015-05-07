package jonnymatts.facerecognition;

import java.util.List;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;

public class KNNHandler implements ProjectClassifier {

	private Classifier classifier;
	private int numberOfNeighbours;
	
	KNNHandler(int n) {
		numberOfNeighbours = n;
	}
	
	public String getClassifierName() {
		return "_KNN_" + numberOfNeighbours;
	}
	
	public void trainForBiometric(PersonDataset trainingSet, Biometric biometric) {
		Dataset data = trainingSet.createKNNDatasetForBiometric(biometric);
		classifier = new KNearestNeighbors(numberOfNeighbours);
		classifier.buildClassifier(data);
	}
	
	public PersonDataset predictClassesForBiometric(PersonDataset testingSet, Biometric biometric) {
		List<Person> pList = testingSet.getPersonList();
		for (Person p : pList) {
			Instance inst = p.createKNNInstanceForBiometric(biometric);
			Object predictedClassObject = classifier.classify(inst);
			boolean nullObject = predictedClassObject == (null);
			int predictedClassValue = nullObject ? -1 : Integer.valueOf((int)predictedClassObject);
			switch(biometric) {
				case AGE: p.predictedAge = PersonAge.predictedValueOf(predictedClassValue); break;
				case AGEFINE: p.predictedAgeFine = PersonAgeFine.predictedValueOf(predictedClassValue); break;
				case ETHNICITY: p.predictedEthnicity = PersonEthnicity.valueOf(predictedClassValue); break;
				default: p.predictedGender = PersonGender.valueOf(predictedClassValue); break;
			}
		}
		return new PersonDataset(testingSet.getName(), pList);
	}
}
