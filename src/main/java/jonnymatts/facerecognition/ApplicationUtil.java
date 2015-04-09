package jonnymatts.facerecognition;

import static java.lang.Integer.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class ApplicationUtil {
	
	public static List<Person> readDataset(String pathToDataset) throws IOException {
		List<Person> personList = new ArrayList<Person>();
		String dir = System.getProperty("user.dir");
		Stream<String> lines = Files.lines(Paths.get(dir + pathToDataset));
		lines.forEach(s -> {
			String[] data = s.split(",");
			Person p = new Person(parseInt(data[0]), parseInt(data[1]), parseInt(data[2]), data[3], data[4]);
			personList.add(p);
			});
		lines.close();
		return personList;
	}
}
