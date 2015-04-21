package jonnymatts.facerecognition;

public class CouldNotFindFaceException extends Exception {

	private static final long serialVersionUID = 4345214950206814374L;

	public CouldNotFindFaceException() {}

    public CouldNotFindFaceException(String message)
    {
       super(message);
    }
}
