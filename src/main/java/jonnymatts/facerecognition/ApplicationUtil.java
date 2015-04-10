package jonnymatts.facerecognition;

import static java.lang.Integer.*;

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

import com.google.common.collect.Lists;

public class ApplicationUtil {
	
	public static final String userDir = System.getProperty("user.dir");
	
	// A function that flattens a list of lists of integers, into a single list of integers
	public static List<Double> flattenList(List<List<Double>> inList) {
		List<Double> result = new ArrayList<Double>();
		for (int i = 0; i < inList.size(); i++) {

			// Recursively call this function
			result.addAll(inList.get(i));
		}
		return result;
	}

	public static PersonDataset performRISEFeatureExtractionOnDataset(PersonDataset ds, double pixelsPerHOGCell) {
		List<Person> personList = ds.getPersonList();
		RISEHandler rh = new RISEHandler(3, 15, 1, (personList.get(0).colourImage.rows() / 10), 0.02, 15, 230, 30, pixelsPerHOGCell);
		for(Person p : personList) {
			List<List<List<Double>>> featureVector = rh.findFeatureVector(p.colourImage, p.depthImage);
			List<Double> fv = flattenList(featureVector.stream().map(l -> flattenList(l)).collect(Collectors.toList()));
			p.setFeatureVector(fv);
		}
		ds.setPersonList(personList);
		return ds;
	}
	
	public static void writeResultSetToFileForKNNClassifier(PersonDataset ds) throws IOException {
		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date());
		String fileName = ds.getName() + "_" + timeStamp + ".data";
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(userDir + "/resources/classifier_inputs/knn/" + fileName)));
		for(Person p : ds.getPersonList()) {
			String personString = p.gender.toString() + "," + p.getFeatureVector().toString().replaceAll("\\[|\\]", "");
			bw.write(personString);
			bw.newLine();
		}
		bw.close();
	}
	
	public static PersonDataset readDataset(String pathToDataset) throws IOException {
		List<Person> personList = new ArrayList<Person>();
		List<String> lines =  Lists.newArrayList(Files.lines(Paths.get(userDir + pathToDataset)).iterator());
		String name = lines.get(0);
		lines.remove(0);
		lines.forEach(s -> {
			String[] data = s.split(",");
			Person p = new Person(parseInt(data[0]), parseInt(data[1]), parseInt(data[2]), data[3], data[4]);
			personList.add(p);
			});
		return new PersonDataset(name, personList);
	}
}
