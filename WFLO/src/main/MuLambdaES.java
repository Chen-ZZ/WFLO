package main;

import java.util.ArrayList;
import java.util.Random;

/**
 * MuLambdaES employs Mu-Lambda Evolutionary Strategy to optimize wind farm layout for certain given wind
 * scenarios. In short, Mu is the number of parents which survive, and Lambda is the number of kids that the μ
 * parents make in total. Notice that Lambda should be a multiple of Mu. ES practitioners usually refer to
 * their algorithm by the choice of Mu & Lambda. For example, if Mu = 5 and λ = 20, then we have a (5, 20) -
 * ES.
 * 
 * @author zhengchen
 * @version 1.0
 */

public class MuLambdaES {

	static final String _CROSSOVER = "crossover";
	static final String _MUTATE = "mutate";

	WindFarmLayoutEvaluator wfle;
	ArrayList<double[][]> populations;
	double[] fitnesses;
	double bestFitness;
	Random random;
	int num_Turbines; // number of turbines in the wind farm.
	int maxEvaluations;
	int num_pop; // number of population.
	int mu; // number of parents selected.
	int lambda; // number of population.
	double mutation_rate;

	String operatorFlag;
	String dataFormat; // the format of turbine coordinates: raw, polar or random projection.

	// wind farm parameters
	double farmHeight;
	double farmWidth;
	double interval;
	double minDistance;
	double blockSize; // the size of each block is 1km x 1km.
	int block_X;
	int block_Y;

	int[][] blocks;

	public MuLambdaES(WindFarmLayoutEvaluator evaluator) {
		wfle = evaluator;
		random = new Random();
		bestFitness = Double.MAX_VALUE;
		blockSize = 1000;
	}

	private void evaluate() {

		double minFitness = Double.MAX_VALUE;

		for (int p = 0; p < populations.size(); p++) {
			double[][] layout = populations.get(p);

			double coe;
			if (wfle.checkConstraint(layout)) {
				wfle.evaluate(layout);
				coe = wfle.getEnergyCost();
			} else {
				coe = Double.MAX_VALUE;
			}

			fitnesses[p] = coe;
			if (fitnesses[p] < minFitness) {
				minFitness = fitnesses[p];
			}

			if (minFitness < bestFitness) {
				bestFitness = minFitness;
				System.out.println(bestFitness);
			}

		}

		// System.out.println(wfle.getNumberOfEvaluation());
	}

	private void initialize() {

		populations = new ArrayList<double[][]>();

		fitnesses = new double[lambda + mu];

		farmWidth = wfle.getFarmWidth();
		farmHeight = wfle.getFarmHeight();
		interval = 8.001 * wfle.getTurbineRadius();
		minDistance = wfle.getMinDistance();

		block_X = (int) (farmWidth / blockSize);
		block_Y = (int) (farmHeight / blockSize);
		blocks = new int[block_X][block_Y];

		int num_Blocks = block_X * block_Y;

		for (int j = 0; j < block_Y; j++) {
			for (int i = block_X - 1; i >= 0; i--) {
				// System.out.println("i:" + i);
				// System.out.println("j:" + j);
				// System.out.println("index:" + num_Blocks);
				blocks[i][j] = num_Blocks;
				num_Blocks--;
			}
		}

		System.out.println("Famr Width:" + farmWidth);
		System.out.println("Famr Height:" + farmHeight);
		// System.out.println("Famr Turbine Interval:" + interval);
		// System.out.println("Famr Turbine Min Distance:" + minDistance);

		// initialize populations, should be done in an incremental way.
		for (int p = 0; p < lambda; p++) {
			double[][] layout = new double[num_Turbines][2];

			for (int t = 0; t < num_Turbines; t++) {

				while (!validateDistance(layout, t)) {
					layout[t - 1] = nextLocation();
				}

				layout[t] = nextLocation();
			}

			// Adjust the very last turbine
			if (layout.length == num_Turbines) {

				while (!validateDistance(layout, layout.length - 1)) {
					layout[layout.length - 1] = nextLocation();
				}
			}
			// System.out.println(wfle.checkConstraint(layout));

			populations.add(layout);

		}
	}

	/**
	 * @author Chen
	 * @return A valid random new location.
	 */
	private double[] nextLocation() {

		double x;
		double y;

		do {
			x = random.nextDouble() * farmWidth;// Random turbine location x within width range.
			y = random.nextDouble() * farmHeight;// Random turbine location y within height range.
		} while (!validateLocation(x, y));

		double[] newLocation = { x, y };

		return newLocation;
	}

	/**
	 * @author Chen
	 * @param x
	 *            The x-axial position for the turbine location.
	 * @param y
	 *            The y-axial position for the turbine location.
	 * @return Validation result.
	 */
	private boolean validateLocation(double x, double y) {

		boolean valid = true;

		for (int o = 0; o < wfle.getObstacles().length; o++) {
			double[] obs = wfle.getObstacles()[o];
			if (x > obs[0] && y > obs[1] && x < obs[2] && y < obs[3]) {
				valid = false;
			}
		}

		return valid;
	}

	/**
	 * @author Chen
	 * @param layout
	 *            this is the unfinished layout arrangement during initialization.
	 * @param t
	 *            this is the number of turbines already placed in the layout, the rest are zeros.
	 * @return a validation result of the input layout.
	 */
	private boolean validateDistance(double[][] layout, int t) {

		boolean valid = true;

		if (t > 0) {
			for (int i = 0; i < t + 1; i++) {
				for (int j = 0; j < t + 1; j++) {
					if (i != j) {
						// calculate the squared distance between both turbines*
						double dist = (layout[i][0] - layout[j][0]) * (layout[i][0] - layout[j][0])
								+ (layout[i][1] - layout[j][1]) * (layout[i][1] - layout[j][1]);
						// System.out.println("distance:" + dist);
						if (dist < minDistance) {
							// System.out.println("Security distance contraint violated between turbines " + i
							// + " ("
							// + layout[i][0] + ", " + layout[i][1] + ") and " + j + " (" + layout[j][0] + ",
							// "
							// + layout[j][1] + "): " + Math.sqrt(dist) + " < " + Math.sqrt(minDistance));
							valid = false;
						}
					}
				}
			}
		}

		return valid;
	}

	/**
	 * @author Chen
	 * @param layout
	 *            Randomly choose a turbine and move it to a new random location.
	 * @return A slightly modified layout.
	 */
	private double[][] mutateTurbine(double[][] layout) {

		int tempIndex = random.nextInt(num_Turbines - 1);

		do {

			layout[tempIndex] = nextLocation();

		} while (!validateDistance(layout, num_Turbines - 1));

		return layout;
	}

	/**
	 * @author Chen
	 * @param layout
	 *            Randomly choose a block of turbines and copy it to another block, then balance the number of
	 *            turbines.
	 * @return A modified layout.
	 */
	private double[][] mutateBlock(double[][] layout) {

		// Divide the layout into several blocks, then randomly copy one block to replace another.
		int origin_X;
		int origin_Y;
		int target_X;
		int target_Y;

		ArrayList<double[]> originBlock = new ArrayList<>();
		ArrayList<double[]> targetBlock = new ArrayList<>();

		// Convert 2-D layout to an ArrayList<double[]>
		ArrayList<double[]> layoutList = new ArrayList<>();
		for (double[] location : layout) {
			layoutList.add(location);
		}

		// Randomly figure out which block to copy.
		do {
			origin_X = random.nextInt(block_X);
			origin_Y = random.nextInt(block_Y);

			target_X = random.nextInt(block_X);
			target_Y = random.nextInt(block_Y);

		} while (origin_X == target_X && origin_Y == target_Y);

		// According to the determined x & y coordinates, figure out the original and target block.
		for (int x = 0; x < layout.length; x++) {

			if ((origin_X * blockSize < layout[x][0] && layout[x][0] < (origin_X + 1) * blockSize)
					&& (origin_Y * blockSize < layout[x][1] && layout[x][1] < (origin_Y + 1) * blockSize)) {
				originBlock.add(layout[x]);
			}

			if ((target_X * blockSize < layout[x][0] && layout[x][0] < (target_X + 1) * blockSize)
					&& (target_Y * blockSize < layout[x][1] && layout[x][1] < (target_Y + 1) * blockSize)) {
				targetBlock.add(layout[x]);
			}

		}

		// System.out.println("--------- Mutate Block ---------");
		// System.out.println(
		// "OriginBlockIndex:" + blocks[origin_X][origin_Y] + ", How many turbines:" + originBlock.size());
		// System.out.println(
		// "TargetBlockIndex:" + blocks[target_X][target_Y] + ", How many turbines:" + targetBlock.size());

		// Remove the turbines in the target block.
		layoutList.removeAll(targetBlock);

		// System.out.println("After Removal Check:" + wfle.checkConstraint(listToArray(layoutList)));

		// According to the determined x & y coordinates, transfer the original turbines to the target block.
		for (double[] turbine : originBlock) {

			double[] tempT = { 0, 0 };

			// System.out.println("o:" + turbine[0]);
			// System.out.println("o:" + turbine[1]);

			tempT[0] = turbine[0] - origin_X * blockSize + target_X * blockSize;
			tempT[1] = turbine[1] - origin_Y * blockSize + target_Y * blockSize;

			// System.out.println("t:" + tempT[0]);
			// System.out.println("t:" + tempT[1]);

			// Replace the target block with origin block.
			// verify the layout after each turbine is added, in case there is an obstacle or invalid
			// distance.

			layoutList.add(tempT);

			boolean checkFlag = wfle.checkConstraint(listToArray(layoutList));

			// System.out.println("Copied a turbine, Check:" + checkFlag);

			if (!checkFlag) {
				layoutList.remove(tempT);
			}

			checkFlag = wfle.checkConstraint(listToArray(layoutList));

			// System.out.println("Removed a turbine, Check:" + checkFlag);

		}

		// System.out.println("After Copy Check:" + wfle.checkConstraint(listToArray(layoutList)));

		// System.out.println("After Copy Turbine Numbers:" + layoutList.size());

		// According to the fixed total number of turbines, balance the new layout.
		if (layoutList.size() >= num_Turbines) {

			while (layoutList.size() > num_Turbines) {
				layoutList.remove(random.nextInt(num_Turbines));
			}
			layout = listToArray(layoutList);

		} else {

			for (int t = layoutList.size(); t < num_Turbines; t++) {
				boolean flag = true;
				do {
					double[] tempLocation = nextLocation();
					layoutList.add(tempLocation);

					flag = false;

					if (!wfle.checkConstraint(listToArray(layoutList))) {
						layoutList.remove(tempLocation);
						flag = true;
					}

				} while (flag);

				// Adjust the very last turbine
				if (layoutList.size() == num_Turbines) {
					layout = listToArray(layoutList);
					while (!validateDistance(layout, layout.length - 1)) {
						layout[layout.length - 1] = nextLocation();
					}
				}

			}

		}

		// System.out.println("After Balancing Turbine Numbers:" + layoutList.size());

		// System.out.println("Block Copy Complete! Check:" + wfle.checkConstraint(layout));

		return layout;
	}

	private double[][] blockCrossover(double[][] parent_A, double[][] parent_B) {

		double[][] child;

		// double[][] parent_A;
		// double[][] parent_B;
		// do {
		// parent_A = parents.get(random.nextInt(parents.size()));
		// parent_B = parents.get(random.nextInt(parents.size()));
		// } while (parent_A.equals(parent_B));

		int origin_X;
		int origin_Y;
		int target_X;
		int target_Y;
		// do {} while (origin_X == target_X && origin_Y == target_Y); // Not necessary to do so since two
		// layouts.
		origin_X = random.nextInt(block_X);
		origin_Y = random.nextInt(block_Y);

		target_X = random.nextInt(block_X);
		target_Y = random.nextInt(block_Y);

		ArrayList<double[]> originBlock = new ArrayList<>();
		ArrayList<double[]> targetBlock = new ArrayList<>();

		// Convert 2-D layout to an ArrayList<double[]>
		ArrayList<double[]> layoutList_A = new ArrayList<>();
		ArrayList<double[]> layoutList_B = new ArrayList<>();
		for (double[] locationA : parent_A) {
			layoutList_A.add(locationA);
		}
		for (double[] locationB : parent_B) {
			layoutList_B.add(locationB);
		}

		// According to the determined x & y coordinates, figure out the original and target block.
		for (int x = 0; x < num_Turbines; x++) {

			if ((origin_X * blockSize < parent_A[x][0] && parent_A[x][0] < (origin_X + 1) * blockSize)
					&& (origin_Y * blockSize < parent_A[x][1]
							&& parent_A[x][1] < (origin_Y + 1) * blockSize)) {
				originBlock.add(parent_A[x]);
			}

			if ((target_X * blockSize < parent_B[x][0] && parent_B[x][0] < (target_X + 1) * blockSize)
					&& (target_Y * blockSize < parent_B[x][1]
							&& parent_B[x][1] < (target_Y + 1) * blockSize)) {
				targetBlock.add(parent_B[x]);
			}

		}

		layoutList_B.removeAll(targetBlock);

		for (double[] turbine : originBlock) {

			double[] tempT = { 0, 0 };

			tempT[0] = turbine[0] - origin_X * blockSize + target_X * blockSize;
			tempT[1] = turbine[1] - origin_Y * blockSize + target_Y * blockSize;

			layoutList_B.add(tempT);

			boolean checkFlag = wfle.checkConstraint(listToArray(layoutList_B));

			if (!checkFlag) {
				layoutList_B.remove(tempT);
			}

		}

		// Balancing the number of turbines after cross-copy.
		if (layoutList_B.size() >= num_Turbines) {

			while (layoutList_B.size() > num_Turbines) {
				layoutList_B.remove(random.nextInt(num_Turbines));
			}
			child = listToArray(layoutList_B);

		} else {

			for (int t = layoutList_B.size(); t < num_Turbines; t++) {
				boolean flag = true;
				do {
					double[] tempLocation = nextLocation();
					layoutList_B.add(tempLocation);

					flag = false;

					if (!wfle.checkConstraint(listToArray(layoutList_B))) {
						layoutList_B.remove(tempLocation);
						flag = true;
					}

				} while (flag);

				// Adjust the very last turbine
				if (layoutList_B.size() == num_Turbines) {
					child = listToArray(layoutList_B);
					while (!validateDistance(child, child.length - 1)) {
						child[child.length - 1] = nextLocation();
					}
				}

			}

		}

		child = listToArray(layoutList_B);

		return child;
	}

	/**
	 * @author Chen
	 * @param list
	 *            An arraylist which contains the 2-D array.
	 * @return a 2-D double array
	 */
	private double[][] listToArray(ArrayList<double[]> list) {

		int size = list.size();
		double[][] array = new double[size][2];

		for (int i = 0; i < size; i++) {
			array[i] = list.get(i);
		}

		return array;
	}

	public double run_Dot_ES() {

		initialize();

		evaluate();

		do {

			// From now on, lambda-mu evolutionary strategy takes over.
			// Select best mu parents, discard the rest.
			int[] winners = new int[mu];
			int[] competitors = new int[lambda];

			for (int c = 0; c < competitors.length; c++) {
				competitors[c] = c;
				int index = random.nextInt(c + 1);
				int temp = competitors[index];
				competitors[index] = competitors[c];
				competitors[c] = temp;
				// System.out.println(temp);
			}

			for (int t = 0; t < winners.length; t++) {
				int winner = -1;
				double winner_fit = Double.MAX_VALUE;
				for (int c = 0; c < lambda; c++) {
					int competitor = competitors[random.nextInt(lambda)];
					if (fitnesses[competitor] < winner_fit) {
						winner = competitor;
						winner_fit = fitnesses[winner];
						// System.out.println("---");
						// System.out.println(winner);
					}
				}
				winners[t] = winner;
			}

			// System.out.println("C:" + Arrays.toString(competitors));
			// System.out.println("W:" + Arrays.toString(winners));

			// Generate lambda/mu children.
			// Generator new individuals using winners.

			ArrayList<double[][]> temp = populations;
			populations = new ArrayList<>();

			for (int i = 0; i < lambda / mu; i++) {

				for (int j = 0; j < winners.length; j++) {

					// Single Turbine Mutation Operator
					// populations.set(competitors[c],
					// mutateTurbine(populations.get(winners[random.nextInt(mu)])));

					// Block Mutation Operator
					if (operatorFlag.equals(_CROSSOVER)) {
						populations.add(mutateBlock(temp.get(winners[j])));
					}

					// Block Crossover Operator
					if (operatorFlag.equals(_MUTATE)) {
						double[][] temp_Parent_A = temp.get(winners[j]);
						double[][] temp_Parent_B = temp.get(winners[j]);
						populations.add(blockCrossover(temp_Parent_A, temp_Parent_B));
					}
				}

			}

			evaluate();

		} while (wfle.getNumberOfEvaluation() < maxEvaluations);

		return bestFitness;

	}

	public double run_Plus_ES() {

		initialize();

		evaluate();

		do {

			// From now on, lambda-mu evolutionary strategy takes over.
			// Select best mu parents, discard the rest.
			int[] winners = new int[mu];
			int[] competitors = new int[populations.size()];

			for (int c = 0; c < competitors.length; c++) {
				competitors[c] = c;
				int index = random.nextInt(c + 1);
				int temp = competitors[index];
				competitors[index] = competitors[c];
				competitors[c] = temp;
				// System.out.println(temp);
			}

			for (int t = 0; t < winners.length; t++) {
				int winner = -1;
				double winner_fit = Double.MAX_VALUE;
				for (int c = 0; c < lambda; c++) {
					int competitor = competitors[random.nextInt(lambda)];
					if (fitnesses[competitor] < winner_fit) {
						winner = competitor;
						winner_fit = fitnesses[winner];
						// System.out.println("---");
						// System.out.println(winner);
					}
				}
				winners[t] = winner;
			}

			// System.out.println("C:" + Arrays.toString(competitors));
			// System.out.println("W:" + Arrays.toString(winners));

			// Generate lambda/mu children.
			// Generator new individuals using winners.

			ArrayList<double[][]> temp = populations;
			populations = new ArrayList<>();

			// Put the winners back into the population
			for (int winner : winners) {
				populations.add(temp.get(winner));
			}

			for (int i = 0; i < lambda / mu; i++) {

				for (int j = 0; j < winners.length; j++) {

					// Single Turbine Mutation Operator
					// populations.set(competitors[c],
					// mutateTurbine(populations.get(winners[random.nextInt(mu)])));

					// Block Mutation Operator
					if (operatorFlag.equals(_CROSSOVER)) {
						populations.add(mutateBlock(temp.get(winners[j])));
					}

					// Block Crossover Operator
					if (operatorFlag.equals(_MUTATE)) {
						double[][] temp_Parent_A = temp.get(winners[j]);
						double[][] temp_Parent_B = temp.get(winners[j]);
						populations.add(blockCrossover(temp_Parent_A, temp_Parent_B));
					}
				}

			}

			evaluate();

		} while (wfle.getNumberOfEvaluation() < maxEvaluations);

		return bestFitness;

	}

	public int getNum_Turbines() {
		return num_Turbines;
	}

	public void setNum_Turbines(int num_Turbines) {
		this.num_Turbines = num_Turbines;
	}

	public int getMu() {
		return mu;
	}

	public void setMu(int mu) {
		this.mu = mu;
	}

	public int getLambda() {
		return lambda;
	}

	public void setLambda(int lambda) {
		this.lambda = lambda;
	}

	public int getMaxEvaluations() {
		return maxEvaluations;
	}

	public void setMaxEvaluations(int maxEvaluations) {
		this.maxEvaluations = maxEvaluations + wfle.getNumberOfEvaluation();
	}

	public String getOperatorFlag() {
		return operatorFlag;
	}

	public void setOperatorFlag(String operatorFlag) {
		this.operatorFlag = operatorFlag;
	}

}
