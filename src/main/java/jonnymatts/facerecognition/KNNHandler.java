package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ImageHelper.loadFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

public class KNNHandler {

	private Classifier classifier;
	
	KNNHandler(String pathToTrainingData, int numberOfNeighbours) throws IOException {
		Dataset data = FileHandler.loadDataset(loadFile(pathToTrainingData),0, ",");
		classifier = new KNearestNeighbors(numberOfNeighbours);
		classifier.buildClassifier(data);
	}
	
	public List<Boolean> predictClassOfTestData(String pathToTestData) throws IOException {
		List<Boolean> boolList = new ArrayList<Boolean>();
		Dataset testData = FileHandler.loadDataset(loadFile(pathToTestData),0, ",");
		for (Instance inst : testData) {
			Object predictedClassValue = classifier.classify(inst);
			Object realClassValue = inst.classValue();
			boolList.add(predictedClassValue.equals(realClassValue));
		}
		return boolList;
	}
}
