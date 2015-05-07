package jonnymatts.facerecognition;

import static jonnymatts.facerecognition.ApplicationUtil.getTimeInMinutesAndSeconds;

public class DatasetResult {

	public String name;
	public double genderCorrect;
	public double ageCorrect;
	public double ethnicityCorrect;
	public long extractorTime;
	public long trainingTime;
	public long predictingTime;
	
	String getResultString() {
		return "[" + name + "]: genderCorrect: " + genderCorrect + ", ageCorrect: " + ageCorrect + ", ethnicityCorrect: " +
				ethnicityCorrect + ", extractorTime: " + getTimeInMinutesAndSeconds(extractorTime) + ", trainingTime: " + 
				getTimeInMinutesAndSeconds(trainingTime) + ", predictingTime: " + getTimeInMinutesAndSeconds(predictingTime);
	}
	
	DatasetResult(String n, double g, double a, double e) {
		name = n;
		genderCorrect = g;
		ageCorrect = a;
		ethnicityCorrect = e;
	}
}
