package jonnymatts.facerecognition;

public class CouldNotFindNoseException extends Exception {

	private static final long serialVersionUID = 4345214950206814374L;

	public CouldNotFindNoseException() {}

    public CouldNotFindNoseException(String message)
    {
       super(message);
    }
}
