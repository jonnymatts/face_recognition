package jonnymatts.facerecognition;

import java.util.HashMap;
import java.util.Map;

public enum Biometric {
	GENDER(0), AGE(1), ETHNICITY(2), AGEFINE(3);
	
	private int value;
	
	private Biometric(int value) {
		this.value = value;
	}
	
	public int getValue() {return value;}
	
	public static Biometric valueOf(int id) {
		return map.get(id);
	}

	private static Map<Integer, Biometric> map = new HashMap<Integer, Biometric>();

    static {
        for (Biometric exEnum : Biometric.values()) {
            map.put(exEnum.value, exEnum);
        }
    }
}
