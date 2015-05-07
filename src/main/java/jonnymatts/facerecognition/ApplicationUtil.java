package jonnymatts.facerecognition;

import static java.lang.Integer.parseInt;
import static jonnymatts.facerecognition.ImageHelper.preprocessImages;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.opencv.core.Mat;

import com.google.common.collect.Lists;

public class ApplicationUtil {

	public static final String userDir = System.getProperty("user.dir");
	
	public static List<ExperimentData> getDatasetsForExperiment() throws IOException {
		PersonDataset iitdTrain = readDataset("/resources/datasets/IIT-D_fold1_training.txt");
		PersonDataset iitdTest = readDataset("/resources/datasets/IIT-D_fold1_testing.txt");
		PersonDataset eurecomTrain = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting1.txt");
		PersonDataset eurecomTest = readDataset("/resources/datasets/EURECOM_neutral_dataset_sitting2.txt");
		
		String iitdOptName = "IIT-D_optimised"; 
		PersonDatasetAnalytics pda = new PersonDatasetAnalytics(iitdTrain);
		pda.addPeopleToPersonList(iitdTest);
		List<PersonDataset> sets = pda.getOptimizedDatasets(iitdOptName, 200, false);
		PersonDataset iitdTrainOpt = sets.get(0);
		PersonDataset iitdTestOpt = sets.get(1);
		saveDataset(iitdTrainOpt);
		saveDataset(iitdTestOpt);
		
		String eurecomOptName = "EURECOM_optimised";
		pda = new PersonDatasetAnalytics(eurecomTrain);
		pda.addPeopleToPersonList(eurecomTest);
		sets = pda.getOptimizedDatasets(eurecomOptName, 200, false);
		PersonDataset eurecomTrainOpt = sets.get(0);
		PersonDataset eurecomTestOpt = sets.get(1);
		saveDataset(eurecomTrainOpt);
		saveDataset(eurecomTestOpt);
		
		String eurecomOptAfName = "EURECOM_optimised_ageFine";
		pda = new PersonDatasetAnalytics(eurecomTrain);
		pda.addPeopleToPersonList(eurecomTest);
		sets = pda.getOptimizedDatasets(eurecomOptAfName, 200, true);
		PersonDataset eurecomTrainOptAf = sets.get(0);
		PersonDataset eurecomTestOptAf = sets.get(1);
		saveDataset(eurecomTrainOptAf);
		saveDataset(eurecomTestOptAf);
		
		String fullCombName = "fully_combined_IIT-D_EURECOM";
		pda = new PersonDatasetAnalytics(iitdTrain);
		pda.addPeopleToPersonList(iitdTest);
		pda.addPeopleToPersonList(eurecomTrain);
		pda.addPeopleToPersonList(eurecomTest);
		sets = pda.getOptimizedDatasets(fullCombName, 200, false);
		PersonDataset fullCombinedTrain = sets.get(0);
		PersonDataset fullCombinedTest = sets.get(1);
		saveDataset(fullCombinedTrain);
		saveDataset(fullCombinedTest);
		
		String optCombName = "optimised_combined_IIT-D_EURECOM";
		pda = new PersonDatasetAnalytics(iitdTrainOpt);
		pda.addPeopleToPersonList(eurecomTrainOpt);
		sets = pda.getOptimizedDatasets("optimised_combined_IIT-D_EURECOM", 200, false);
		PersonDataset optCombinedTrain = sets.get(0);
		PersonDataset optCombinedTest = sets.get(1);
		saveDataset(optCombinedTrain);
		saveDataset(optCombinedTest);
		
		List<ExperimentData> dataList = Arrays.asList(new ExperimentData("IIT-D", iitdTrain, iitdTest, false),
													  new ExperimentData("EURECOM", eurecomTrain, eurecomTest, false),
													  new ExperimentData("EURECOM_ageFine", eurecomTrain, eurecomTest, true),
													  new ExperimentData(iitdOptName, iitdTrainOpt, iitdTestOpt, false),
													  new ExperimentData(eurecomOptName, eurecomTrainOpt, eurecomTestOpt, false),
													  new ExperimentData(eurecomOptAfName, eurecomTrainOptAf, eurecomTestOptAf, true),
													  new ExperimentData(fullCombName, fullCombinedTrain, fullCombinedTest, false),
													  new ExperimentData(optCombName, optCombinedTrain, optCombinedTest, false));
		return dataList;
	}
	
	public static String getTimeInMinutesAndSeconds(long millis){
		return String.format("%d min, %d sec", 
		    TimeUnit.MILLISECONDS.toMinutes(millis),
		    TimeUnit.MILLISECONDS.toSeconds(millis) - 
		    TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(millis))
		);
	}
	
	public static List<DatasetResult> runFeatureExtractionTestsForGivenExtractor(String name, PersonDataset trainingSet,
			PersonDataset testingSet, ProjectFeatureExtractor fe, List<ProjectClassifier> classifiers,
			boolean sampleFeatureVector, boolean useAgeFine) throws IOException {
		
		List<DatasetResult> resultSets = new ArrayList<DatasetResult>();
		
		long startTime = System.currentTimeMillis();
		
		// Perform feature extraction on both datasets		
		PersonDataset eTrainingSet = fe.performFeatureExtractionForDataset(trainingSet, sampleFeatureVector);
		PersonDataset eTestingSet = fe.performFeatureExtractionForDataset(testingSet, sampleFeatureVector);
		
		long endTime = System.currentTimeMillis();
		long featureExtractorTime = endTime - startTime;

		
		for(ProjectClassifier classifier : classifiers) {
			
			// Train the classifer, then predict the class of testing data for each biometric
			startTime = System.currentTimeMillis();
			classifier.trainForBiometric(eTrainingSet, Biometric.GENDER);
			endTime = System.currentTimeMillis();
			long trainTime = endTime - startTime;
			
			startTime = System.currentTimeMillis();
			PersonDataset resultSet = classifier.predictClassesForBiometric(eTestingSet, Biometric.GENDER);
			endTime = System.currentTimeMillis();
			long predictTime = endTime - startTime;

			if(useAgeFine) {
				classifier.trainForBiometric(eTrainingSet, Biometric.AGEFINE);
				resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.AGEFINE);
			} else {
				classifier.trainForBiometric(eTrainingSet, Biometric.AGE);
				resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.AGE);
			}

			classifier.trainForBiometric(eTrainingSet, Biometric.ETHNICITY);
			resultSet = classifier.predictClassesForBiometric(resultSet, Biometric.ETHNICITY);

			// Write experiment output to file
			String resultsName = name + fe.getExtractorName() + classifier.getClassifierName();
			if(sampleFeatureVector) resultsName += "_sampled";
			resultSet.setName(resultsName);
			saveResultSet(resultSet.getName(), resultSet, useAgeFine);
			resultSets.add(resultSet.getDatasetResult(featureExtractorTime, trainTime, predictTime, useAgeFine));
		}
		return resultSets;
	}
	
	public static void runFeatureExtractionTestsOnDatasets(String name, PersonDataset trainingSet, PersonDataset testingSet,
			boolean sampleFeatureVector, boolean useAgeFine) throws IOException {
		List<DatasetResult> resultSets = new ArrayList<DatasetResult>();
		
		long totalStartTime = System.currentTimeMillis();
		
		// Pre-process all the images
		trainingSet.setPersonList(performPreprocessing(trainingSet, 128));
		testingSet.setPersonList(performPreprocessing(testingSet, 128));
		
		// Initialise all the feature extractors
		LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler(8, 1, true, false, false, 5);
		EigenfaceHandler efh = new EigenfaceHandler(trainingSet);
		RISEHandler rh = new RISEHandler(3, 15, 1, (trainingSet.getPersonList().get(0).colourImage.rows() / 10), 0.02, 15,
				230, 30, 25);
		GradientLBPHandler glbph = new GradientLBPHandler(8);
		
		// Initialise all the classifiers
		ProjectClassifier knn1 = new KNNHandler(1);
		ProjectClassifier knn3 = new KNNHandler(3);
		ProjectClassifier svm = new SVMHandler();
		List<ProjectClassifier> classifiers = Arrays.asList(knn1, knn3, svm);
		
		
		String reportName = sampleFeatureVector ? name + "_sampled" : name;
		// Run all the feature extraction tests
		long startTime = System.currentTimeMillis();
		System.out.println("[" + reportName + "] Running LBP tests...");
		resultSets.addAll(runFeatureExtractionTestsForGivenExtractor(name, trainingSet, testingSet, lbph, classifiers, sampleFeatureVector, useAgeFine));
		long endTime = System.currentTimeMillis();
		System.out.println("[" + reportName + "] LBP tests took " + getTimeInMinutesAndSeconds(endTime - startTime) + ". Running RGB-LBP tests...");
		
		lbph.setUseRGB(true);
		startTime = System.currentTimeMillis();
		resultSets.addAll(runFeatureExtractionTestsForGivenExtractor(name, trainingSet, testingSet, lbph, classifiers, sampleFeatureVector, useAgeFine));
		endTime = System.currentTimeMillis();
		System.out.println("[" + reportName + "] RGB-LBP tests took " + getTimeInMinutesAndSeconds(endTime - startTime) + ". Running Eigenface tests...");
		
		startTime = System.currentTimeMillis();
		resultSets.addAll(runFeatureExtractionTestsForGivenExtractor(name, trainingSet, testingSet, efh, classifiers, sampleFeatureVector, useAgeFine));
		endTime = System.currentTimeMillis();
		System.out.println("[" + reportName + "] Eigenface tests took " + getTimeInMinutesAndSeconds(endTime - startTime) + ". Running GLBP tests...");
		
		startTime = System.currentTimeMillis();
		resultSets.addAll(runFeatureExtractionTestsForGivenExtractor(name, trainingSet, testingSet, glbph, classifiers, sampleFeatureVector, useAgeFine));
		endTime = System.currentTimeMillis();
		System.out.println("[" + reportName + "] GLBP tests took " + getTimeInMinutesAndSeconds(endTime - startTime) + ". Running RISE tests...");
		
		// Resize images for RISEHandler
		trainingSet.setPersonList(trainingSet.resizeImages(256));
		testingSet.setPersonList(testingSet.resizeImages(256));
		
		startTime = System.currentTimeMillis();
		resultSets.addAll(runFeatureExtractionTestsForGivenExtractor(name, trainingSet, testingSet, rh, classifiers, sampleFeatureVector, useAgeFine));
		endTime = System.currentTimeMillis();
		System.out.println("[" + reportName + "] RISE tests took " + getTimeInMinutesAndSeconds(endTime - startTime));
		
		long totalEndTime = System.currentTimeMillis();
		
		saveTestReport(reportName, resultSets, (totalEndTime - totalStartTime));
	}
	
	public static String createIITDDatasetString(List<String> l, int i) {
		String returnString = l.get(0);
		returnString += "," + l.get(1);
		returnString += "," + l.get(2);
		returnString += ",/resources/face_testing_images/IIIT-D Kinect RGB-D Face Database/fold1/testing/" + i + "/RGB/" + l.get(3) + ".jpg";
		returnString += ",/resources/face_testing_images/IIIT-D Kinect RGB-D Face Database/fold1/testing/" + i + "/Depth/" + l.get(3) + ".jpg";
		return returnString;
	}
	
	public static DatasetResult averageDatasetResultList(List<DatasetResult> inputList) {
		 double genderSum = inputList.stream().map(dr -> dr.genderCorrect).reduce(0d, Double::sum);
		 double ageSum = inputList.stream().map(dr -> dr.ageCorrect).reduce(0d, Double::sum);
		 double ethnicitySum = inputList.stream().map(dr -> dr.ethnicityCorrect).reduce(0d, Double::sum);
		 int listSize = inputList.size();
		 return new DatasetResult("averaged_results", (genderSum / listSize), (ageSum / listSize), (ethnicitySum / listSize));
	}
	
	public static String createCommaSeperatedString(List<String> inputList) {
		String returnString = "";
		for(int i = 0; i < inputList.size(); i++) {
			String addString = (i == (inputList.size() -1)) ? inputList.get(i) : (inputList.get(i) + ",");
			returnString += addString;
		}
		return returnString;
	}
	
	public static String getBiometricString(Biometric biometric) {
		switch (biometric) {
			case AGE: return "_age_";
			case AGEFINE: return "_ageFine_";
			case ETHNICITY: return "_ethnicity_";
			default: return "_gender_";
		}
	}
	
	public static String getTimestamp() {
		return new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date());
	}

	// A function that flattens a list of lists of integers, into a single list
	// of integers
	public static List<Double> flattenList(List<List<Double>> inList) {
		List<Double> result = new ArrayList<Double>();
		for (int i = 0; i < inList.size(); i++) {

			// Recursively call this function
			result.addAll(inList.get(i));
		}
		return result;
	}

	public static double findPercentageCorrect(List<Boolean> boolList) {
		double trueCount = 0d;
		for (Boolean b : boolList) {
			if (b == true)
				trueCount++;
		}
		return (trueCount / boolList.size());
	}

	public static List<Person> performPreprocessing(PersonDataset set, int dimension) {
		List<Person> pList = set.getPersonList();
		for (Person p : pList) {
			List<Mat> processedImages = preprocessImages(p.colourImage, p.depthImage, dimension);
			if (!processedImages.isEmpty()) {
				p.setIsPreprocessed(true);
				p.colourImage = processedImages.get(0);
				p.depthImage = processedImages.get(1);
			}
		}
		return pList;
	}

	public static PersonDataset performGLBPFeatureExtractionOnDataset(PersonDataset ds, int population) {
		List<Person> personList = ds.getPersonList();
		GradientLBPHandler glbph = new GradientLBPHandler(population);
		for (Person p : personList) {
			List<List<Double>> featureVector = glbph.findFeatureVector(p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static PersonDataset performRISEFeatureExtractionOnDataset(PersonDataset ds, double pixelsPerHOGCell) {
		List<Person> personList = ds.getPersonList();
		RISEHandler rh = new RISEHandler(3, 15, 1, (personList.get(0).colourImage.rows() / 10), 0.02, 15, 230, 30,
				pixelsPerHOGCell);
		for (Person p : personList) {
			List<List<List<Double>>> featureVector = rh.findFeatureVector(p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector.stream().map(l -> flattenList(l)).collect(Collectors.toList()));
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static List<PersonDataset> performEigenfaceFeatureExtractionOnDataset(PersonDataset trainingSet, PersonDataset testingSet) {
		List<Person> trainingPersonList = trainingSet.getPersonList();
		EigenfaceHandler efh = new EigenfaceHandler(trainingSet);
		for (Person p : trainingPersonList) {
			List<List<Double>> featureVector = efh.findFeatureVector(p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		List<Person> testingPersonList = testingSet.getPersonList();
		for (Person p : testingPersonList) {
			List<List<Double>> featureVector = efh.findFeatureVector(p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		trainingSet.setPersonList(trainingPersonList);
		testingSet.setPersonList(testingPersonList);
		return Arrays.asList(trainingSet, testingSet);
	}

	public static PersonDataset performLBPFeatureExtractionOnDataset(PersonDataset ds, int noOfSubImages, int population, 
			int radius, boolean useUniformPatterns, boolean useRotationInvariance, boolean useRGB) {
		List<Person> personList = ds.getPersonList();
		LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler(population, radius, useUniformPatterns,
				useRotationInvariance, useRGB, noOfSubImages);
		for (Person p : personList) {
			List<List<Double>> featureVector = lbph.findFeatureVector(p.colourImage);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static void writeResultSetToFilesForKNNClassifier(PersonDataset ds, String extractionMethod) throws IOException {
		writeResultSetToFileForKNNClassifier(ds, extractionMethod, Biometric.GENDER);
		writeResultSetToFileForKNNClassifier(ds, extractionMethod, Biometric.AGE);
		writeResultSetToFileForKNNClassifier(ds, extractionMethod, Biometric.ETHNICITY);
	}

	public static void writeResultSetToFileForKNNClassifier(PersonDataset ds,
			String extractionMethod, Biometric biometric) throws IOException {
		String biometricString = getBiometricString(biometric);
		String fileName = ds.getName() + biometricString + extractionMethod
				+ "_" + getTimestamp() + ".data";
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(userDir
				+ "/resources/classifier_inputs/knn/" + fileName)));
		for (Person p : ds.getPersonList()) {
			String personBiometric;
			switch (biometric) {
			case AGE:
				personBiometric = p.age.toString();
				break;
			case ETHNICITY:
				personBiometric = p.ethnicity.toString();
				break;
			default:
				personBiometric = p.gender.toString();
				break;
			}
			String personString = personBiometric + ", "
					+ p.getFeatureVector().toString().replaceAll("\\[|\\]", "");
			bw.write(personString);
			bw.newLine();
		}
		bw.close();
	}

	public static void writeResultSetToFileForSVMClassifier(PersonDataset ds,
			String extractionMethod, Biometric biometric) throws IOException {
		String biometricString = getBiometricString(biometric);
		String fileName = ds.getName() + biometricString + extractionMethod
				+ "_" + getTimestamp() + ".data";
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(userDir
				+ "/resources/classifier_inputs/svm/" + fileName)));
		for (Person p : ds.getPersonList()) {
			int personBiometric;
			switch (biometric) {
			case AGE:
				personBiometric = p.age.getValue();
				break;
			case ETHNICITY:
				personBiometric = p.ethnicity.getValue();
				break;
			default:
				personBiometric = p.gender.getValue();
				break;
			}
			String personString = personBiometric + " ";
			List<Double> fv = p.getFeatureVector();
			for (int i = 0; i < fv.size(); i++) {
				personString = personString + (i + 1) + ":" + fv.get(i) + " ";
			}
			bw.write(personString);
			bw.newLine();
		}
		bw.close();
	}
	
	public static void saveResultSet(String name, PersonDataset ds, boolean useAgeFine) throws IOException {
		String fileName = name;
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(userDir + "/resources/results/" + fileName + ".txt")));
		bw.write(fileName);
		bw.newLine();
		for(Person p : ds.getPersonList()) {
			String genderBoolean = String.valueOf(p.gender == p.predictedGender);
			String ageString = useAgeFine ? p.ageFine.toString() : p.age.toString();
			String agePredictedString = useAgeFine ? p.predictedAgeFine.toString() : p.predictedAge.toString();
			String ageBoolean = useAgeFine ? String.valueOf(p.ageFine == p.predictedAgeFine) : String.valueOf(p.age == p.predictedAge); 
			String ethnicityBoolean = String.valueOf(p.ethnicity == p.predictedEthnicity);
			List<String> personList = Arrays.asList(p.name, p.gender.toString(), p.predictedGender.toString(), genderBoolean,
															ageString, agePredictedString, ageBoolean,
															p.ethnicity.toString(), p.predictedEthnicity.toString(), ethnicityBoolean);
			String personString = createCommaSeperatedString(personList);
			bw.write(personString);
			bw.newLine();
		}
		bw.close();
	}
	
	public static PersonDataset readResultSet(String pathToResultSet)
			throws IOException {
		List<Person> personList = new ArrayList<Person>();
		List<String> lines = Lists.newArrayList(Files.lines(
				Paths.get(userDir + pathToResultSet)).iterator());
		String name = lines.get(0);
		lines.remove(0);
		for(int i = 0; i < lines.size(); i++) {
			String[] data = lines.get(i).split(",");
			Person p = new Person(data[0], data[1], data[2], data[4], data[5], data[7], data[8]);
			personList.add(p);
		}
		return new PersonDataset(name, personList);
	}
	
	public static void saveTestReport(String name, List<DatasetResult> drList, long time) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(userDir + "/resources/test_reports/" + name + ".txt")));
		bw.write(name + " - Test took " + getTimeInMinutesAndSeconds(time));
		bw.newLine();
		for (DatasetResult dr : drList) {
			bw.write(dr.getResultString());
			bw.newLine();
		}
		bw.close();
	}
	
	public static void saveDataset(PersonDataset ds) throws IOException {
		String fileName = ds.getName();
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(userDir + "/resources/datasets/" + fileName + ".txt")));
		bw.write(fileName);
		bw.newLine();
		for (Person p : ds.getPersonList()) {
			String personString = p.gender.getValue() + "," + p.actualAge + "," + p.ethnicity.getValue() + "," + 
					p.colourImagePath + "," + p.depthImagePath; 
			bw.write(personString);
			bw.newLine();
		}
		bw.close();
	}

	public static PersonDataset readDataset(String pathToDataset)
			throws IOException {
		List<Person> personList = new ArrayList<Person>();
		List<String> lines = Lists.newArrayList(Files.lines(
				Paths.get(userDir + pathToDataset)).iterator());
		String name = lines.get(0);
		lines.remove(0);
		for(int i = 0; i < lines.size(); i++) {
			String[] data = lines.get(i).split(",");
			String personName = name + "_" + i;
			Person p = new Person(personName, parseInt(data[0]), parseInt(data[1]),
					parseInt(data[2]), data[3], data[4]);
			personList.add(p);
		}
		return new PersonDataset(name, personList);
	}
}
