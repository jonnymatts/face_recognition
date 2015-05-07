package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.*;

import java.io.IOException;
import java.util.List;

import javax.swing.JOptionPane;

public class Application {
	public static void main(String[] args) throws IOException {
		// Load .dylib file for openCV
		System.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/java/libopencv_java2410.dylib");
		
		List<ExperimentData> dataList = getDatasetsForExperiment();
		
		long startTime = System.currentTimeMillis();
		
		for(ExperimentData ed : dataList) {
			runFeatureExtractionTestsOnDatasets(ed.name, new PersonDataset(ed.trainingSet), new PersonDataset(ed.testingSet), false, ed.useAgeFine);
			runFeatureExtractionTestsOnDatasets(ed.name, new PersonDataset(ed.trainingSet), new PersonDataset(ed.testingSet), true, ed.useAgeFine);
		}
		
		long endTime = System.currentTimeMillis();
		
		JOptionPane.showMessageDialog(null, "Finished running tests: Total time taken - " + 
				getTimeInMinutesAndSeconds(endTime-startTime));
	}
}
