package jonnymatts.facerecognition;

public class DatasetResult {

	public double genderCorrect;
	public double ageCorrect;
	public double ethnicityCorrect;
	
	String getResultString() {
		return "genderCorrect: " + genderCorrect + ", ageCorrect: " + ageCorrect + ", ethnicityCorrect: " + ethnicityCorrect;
	}
	
	DatasetResult(double g, double a, double e) {
		genderCorrect = g;
		ageCorrect = a;
		ethnicityCorrect = e;
	}
}
