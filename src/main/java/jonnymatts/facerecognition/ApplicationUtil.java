package jonnymatts.facerecognition;

import static java.lang.Integer.*;
import static jonnymatts.facerecognition.ImageHelper.preprocessImages;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

import org.opencv.core.Mat;

import com.google.common.collect.Lists;

public class ApplicationUtil {

	public static final String userDir = System.getProperty("user.dir");

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

	public static List<Person> performPreprocessing(PersonDataset set,
			int dimension) {
		List<Person> pList = set.getPersonList();
		for (Person p : pList) {
			List<Mat> processedImages = preprocessImages(p.colourImage,
					p.depthImage, 256);
			if (!processedImages.isEmpty()) {
				p.setIsPreprocessed(true);
				p.colourImage = processedImages.get(0);
				p.depthImage = processedImages.get(1);
			}
		}
		return pList;
	}

	public static PersonDataset performGLBPFeatureExtractionOnDataset(
			PersonDataset ds, int population, int radius) {
		List<Person> personList = ds.getPersonList();
		GradientLBPHandler glbph = new GradientLBPHandler(population, radius);
		for (Person p : personList) {
			List<List<Double>> featureVector = glbph.findFeatureVector(
					p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static PersonDataset performRISEFeatureExtractionOnDataset(
			PersonDataset ds, double pixelsPerHOGCell) {
		List<Person> personList = ds.getPersonList();
		RISEHandler rh = new RISEHandler(3, 15, 1,
				(personList.get(0).colourImage.rows() / 10), 0.02, 15, 230, 30,
				pixelsPerHOGCell);
		for (Person p : personList) {
			List<List<List<Double>>> featureVector = rh.findFeatureVector(
					p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector.stream()
					.map(l -> flattenList(l)).collect(Collectors.toList()));
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static PersonDataset performEigenfaceFeatureExtractionOnDataset(
			PersonDataset ds) {
		List<Person> personList = ds.getPersonList();
		EigenfaceHandler efh = new EigenfaceHandler(ds.getColourImageList(),
				ds.getDepthImageList());
		for (Person p : personList) {
			List<List<Double>> featureVector = efh.findFeatureVector(
					p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static PersonDataset perfromLBPFeatureExtractionOnDataset(PersonDataset ds, int noOfSubImages, int population, 
			int radius, boolean useUniformPatterns, boolean useRotationInvariance, boolean useRGB) {
		List<Person> personList = ds.getPersonList();
		LocalBinaryPatternHandler lbph = new LocalBinaryPatternHandler(population, radius, useUniformPatterns,
				useRotationInvariance, useRGB);
		for (Person p : personList) {
			List<List<Double>> featureVector = lbph.findFeatureVector(p.colourImage, noOfSubImages);
			List<Double> fv = flattenList(featureVector);
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}

	public static void writeResultSetToFilesForKNNClassifier(PersonDataset ds,
			String extractionMethod) throws IOException {
		writeResultSetToFileForKNNClassifier(ds, extractionMethod,
				Biometric.GENDER);
		writeResultSetToFileForKNNClassifier(ds, extractionMethod,
				Biometric.AGE);
		writeResultSetToFileForKNNClassifier(ds, extractionMethod,
				Biometric.ETHNICITY);
	}

	public static void writeResultSetToFileForKNNClassifier(PersonDataset ds,
			String extractionMethod, Biometric biometric) throws IOException {
		String biometricString;
		switch (biometric) {
		case AGE:
			biometricString = "_age_";
			break;
		case ETHNICITY:
			biometricString = "_ethnicity_";
			break;
		default:
			biometricString = "_gender_";
			break;
		}
		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss")
				.format(new Date());
		String fileName = ds.getName() + biometricString + extractionMethod
				+ "_" + timeStamp + ".data";
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
		String biometricString;
		switch (biometric) {
		case AGE:
			biometricString = "_age_";
			break;
		case ETHNICITY:
			biometricString = "_ethnicity_";
			break;
		default:
			biometricString = "_gender_";
			break;
		}
		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss")
				.format(new Date());
		String fileName = ds.getName() + biometricString + extractionMethod
				+ "_" + timeStamp + ".data";
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
