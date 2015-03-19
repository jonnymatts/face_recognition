package jonnymatts.facerecognition;

public enum LBPColour {
	GREY(0), BLUE(0), GREEN(1), RED(2);
	
	private int value;
	
	private LBPColour(int value) {
		this.value = value;
	}
	
	public int getValue() {return value;}
}
