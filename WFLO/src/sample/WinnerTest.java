package sample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class WinnerTest {

	public static void main(String[] args) {

		int lambda = 12;
		int lambda_star = lambda * 2;
		Random random = new Random();

		double[] fitnesses = { 0.0014051052535391845, 0.0014050569669934276, 0.00140417975680632,
				0.00140417975680632, 0.0014040228977188509, 0.0014043215150457342, 0.001403614396701179,
				0.0014050491851560951, 0.0014037973446829218, 0.0014058297235913304, 0.001404890213486524,
				0.001406484126814014, 0.001407258002407732, 0.0014043386949415936, 0.0014045867482417552,
				0.0014072021129629974, 0.001403475807665698, 0.0014051146717247711, 0.0014056878763748695,
				0.0014083658888230316, 0.001403767267649218, 0.0014035418951376837, 0.0014045088599889983,
				0.0014051924996410323 };

		System.out.println(Arrays.toString(fitnesses));

		ArrayList<Double> temp = new ArrayList<>();
		for (Double fitness : fitnesses) {
			temp.add(fitness);
		}

		Arrays.sort(fitnesses);

		for (int i = 0; i < 12; i++) {
			System.out.println(temp.indexOf(fitnesses[i]));
		}

		System.out.println(Arrays.toString(fitnesses));
	}
}
