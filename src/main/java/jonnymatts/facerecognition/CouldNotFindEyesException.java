package jonnymatts.facerecognition;

public class CouldNotFindEyesException extends Exception {

	private static final long serialVersionUID = 4345214950206814374L;

	public CouldNotFindEyesException() {}

    public CouldNotFindEyesException(String message)
    {
       super(message);
    }
}
