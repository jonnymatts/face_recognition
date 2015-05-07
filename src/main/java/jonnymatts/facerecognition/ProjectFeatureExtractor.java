package jonnymatts.facerecognition;

public interface ProjectFeatureExtractor {

	PersonDataset performFeatureExtractionForDataset(PersonDataset trainingSet, boolean sample);

	String getExtractorName();

}
