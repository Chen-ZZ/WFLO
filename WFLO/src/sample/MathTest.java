package sample;

public class MathTest {

	public static void main(String[] args) {
		double x = 1;
		double y = 1;

		double d = Math.hypot(x, y);
		double theta = Math.atan2(y, x);

		System.out.println(d + " " + Math.toDegrees(theta));
	}

}
