package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import my.weka.MyDenseInstance;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * MuLambdaES employs Mu-Lambda Evolutionary Strategy to optimize wind farm layout for certain given wind
 * scenarios. In short, Mu is the number of parents which survive, and Lambda is the number of kids that the μ
 * parents make in total. Notice that Lambda should be a multiple of Mu. ES practitioners usually refer to
 * their algorithm by the choice of Mu & Lambda. For example, if Mu = 5 and λ = 20, then we have a (5, 20) -
 * ES.
 * 
 * @author zhengchen
 * @version 1.6
 */

public class MuLambdaESML {

	static final String _CROSSOVER = "crossover";
	static final String _MUTATE = "mutate";
	String operatorFlag; // for crossover breeding or mutate breeding.

	// layout array with energy cost.
	HashMap<double[][], Double> layoutsData;
	Classifier classifier;
	FastVector attributes;
	Instances trainData;

	WindFarmLayoutEvaluator wfle;
	ArrayList<double[][]> populations; // Array list to store the entire population.
	ArrayList<double[][]> lambda_star_winners; // Array list to store the lambda_star winners.
	double[] fitnesses;
	double[] fitnesses_best;
	double bestFitness;
	double[][] bestLayout; // To store the best layout that have found so far.

	TreeMap<Double, double[][]> FitnessToLayout_Map; // A map to store Layout-to-Fitness.

	Random random;
	int num_Turbines; // number of turbines in the wind farm.
	int maxEvaluations;
	int trainEvaluations;
	int num_pop; // number of population.
	int mu; // number of parents selected.
	int lambda; // number of population.
	int lambda_star; // number of individual selection strategies (for Best & Pre).

	String dataFormat; // raw data format or polar data format.

	// wind farm parameters
	double farmHeight;
	double farmWidth;
	double interval;
	double minDistance;
	double blockSize; // the size of each block is approximately 1km x 1km.
	int block_X;
	int block_Y;

	int[][] blocks;

	public MuLambdaESML(WindFarmLayoutEvaluator evaluator) {
		wfle = evaluator;
		random = new Random();
		bestFitness = Double.MAX_VALUE;
		blockSize = 1000;
	}

	public void evaluateES() {

		double minFitness = Double.MAX_VALUE;

		for (int p = 0; p < lambda; p++) {
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
				// layoutsData.put(layout, coe);
				System.out.println(bestFitness);
			}

			if (layoutsData.size() < trainEvaluations && !layoutsData.containsKey(layout)) {
				layoutsData.put(layout, coe);
			}

		}

	}

	public void evaluate_ML() {

		double minFitness = Double.MAX_VALUE;

		for (int p = 0; p < populations.size(); p++) {
			double[][] layout = populations.get(p);

			double coe_actual;
			double coe_predicted;

			if (wfle.checkConstraint(layout)) {
				coe_predicted = predictCoE(layout);
			} else {
				coe_predicted = Double.MAX_VALUE;
			}

			fitnesses[p] = coe_predicted;
			if (fitnesses[p] < minFitness) {

				wfle.evaluate(layout);
				coe_actual = wfle.getEnergyCost();

				if (coe_actual < coe_predicted) {
					fitnesses[p] = coe_actual;
				}

				minFitness = fitnesses[p];
			}

			if (minFitness < bestFitness) {
				bestFitness = minFitness;
				System.out.println(bestFitness);
			}

		}

	}

	public void initialize() {

		// Set up Attributes
		attributes = new FastVector();
		for (int i = 0; i < num_Turbines; i++) {
			attributes.addElement(new Attribute("d_" + i));
			attributes.addElement(new Attribute("theta_" + i));
		}
		attributes.addElement(new Attribute("cost_of_energy"));

		// Create instances

		trainData = new Instances("LayoutDataSet", attributes, 0);
		trainData.setClassIndex(trainData.numAttributes() - 1);

		layoutsData = new HashMap<double[][], Double>();

		populations = new ArrayList<double[][]>();
		fitnesses = new double[lambda];

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

	public void breeding() {

		// From now on, lambda-mu evolutionary strategy takes over.
		// Select best mu parents, discard the rest.
		ArrayList<Double> tempFitnesses = new ArrayList<>();
		for (Double fitness : fitnesses) {
			tempFitnesses.add(fitness);
		}

		Arrays.sort(fitnesses);

		int[] winners = new int[mu];
		for (int i = 0; i < winners.length; i++) {
			winners[i] = tempFitnesses.indexOf(fitnesses[i]);
		}

		// System.out.println(Arrays.toString(competitors));
		// System.out.println(Arrays.toString(winners));

		ArrayList<double[][]> temp = populations;
		populations = new ArrayList<>();

		// Add the winner back into the population, so a better solution might survive lots of generations.
		// for (int index : winners) {
		// populations.add(temp.get(index));
		// }

		// Generate lambda/mu children.
		// Generator new individuals using winners.
		for (int i = 0; i < lambda / mu; i++) {

			for (int j = 0; j < winners.length; j++) {

				// Block Mutation Operator
				if (operatorFlag.equals(_MUTATE)) {
					populations.add(mutateBlock(temp.get(winners[j])));
				}

				// Block Crossover Operator
				if (operatorFlag.equals(_CROSSOVER)) {
					double[][] temp_Parent_A = temp.get(winners[j]);
					double[][] temp_Parent_B = temp.get(winners[j]);
					populations.add(blockCrossover(temp_Parent_A, temp_Parent_B));
				}
			}

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
	public double[][] mutateBlock(double[][] layout) {

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

	public double[][] blockCrossover(double[][] parent_A, double[][] parent_B) {

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
	public double[][] listToArray(ArrayList<double[]> list) {

		int size = list.size();
		double[][] array = new double[size][2];

		for (int i = 0; i < size; i++) {
			array[i] = list.get(i);
		}

		return array;
	}

	/**
	 * @author Chen
	 * @param layout
	 *            layout data in Raw 2-D array formmat.
	 * @return weka instance
	 */
	private Instance toInstance_Raw(double[][] layout) {

		List<Double> tempValues = new ArrayList<Double>();

		for (double[] coordinates : layout) {
			tempValues.add(coordinates[0]);
			tempValues.add(coordinates[1]);
		}
		// tempValues.add(0.0);// the class value.

		Instance tempInstance = new DenseInstance(1.0,
				tempValues.stream().mapToDouble(Double::doubleValue).toArray());

		tempInstance.setDataset(trainData);

		return tempInstance;
	}

	/**
	 * @author Chen
	 * @param layout
	 *            layout data in Raw 2-D array formmat.
	 * @return weka instance
	 */
	private Instance toInstance_Raw(double[][] layout, double coe) {

		List<Double> tempValues = new ArrayList<Double>();

		for (double[] coordinates : layout) {
			tempValues.add(coordinates[0]);
			tempValues.add(coordinates[1]);
		}
		tempValues.add(coe);// the class value.

		Instance tempInstance = new DenseInstance(1.0,
				tempValues.stream().mapToDouble(Double::doubleValue).toArray());

		tempInstance.setDataset(trainData);

		return tempInstance;
	}

	/**
	 * @author Chen
	 * @param layout
	 *            layout data in Sorted Polar coordinates format.
	 * @return weka instance
	 */
	private Instance toInstance_Polar(double[][] layout, double coe) {

		List<Double> tempValues = new ArrayList<Double>();

		for (double[] coordinates : layout) {
			tempValues.add(Math.hypot(coordinates[0], coordinates[1]));
			tempValues.add(Math.atan2(coordinates[1], coordinates[0]));
		}
		tempValues.add(coe);// the class value.

		Instance tempInstance = new DenseInstance(1.0,
				tempValues.stream().mapToDouble(Double::doubleValue).toArray());

		tempInstance.setDataset(trainData);

		return tempInstance;

	}

	/**
	 * @author Chen A classifier specified by user.
	 */
	public void trainClassifier() {

		// Filling in training data
		for (Map.Entry<double[][], Double> entry : layoutsData.entrySet()) {

			Instance tempInstance;
			// Transform the raw type data into polar coordinates
			if (dataFormat.equals("polar")) {
				tempInstance = toInstance_Polar(entry.getKey(), entry.getValue());
			} else {
				tempInstance = toInstance_Raw(entry.getKey(), entry.getValue());
			}

			tempInstance.setClassValue(entry.getValue());

			trainData.add(tempInstance);
		}

		// If the dataset are in Polar format, then sort all the coordinates according to distance.
		if (dataFormat.equals("polar")) {
			trainData = sortPolarInstances(trainData);
		}

		// Train the classifier
		try {

			classifier.buildClassifier(trainData);
			Evaluation evaluation = new Evaluation(trainData);
			evaluation.evaluateModel(classifier, trainData);
			System.out.println(evaluation.toSummaryString());

			// Use the classifier
			// System.out.println(MLP.classifyInstance(trainData.instance(0)));

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public Instances sortPolarInstances(Instances trainData) {
		System.out.println("--- Sorting Polar Coordinates...");

		double[] tempDistances = new double[(trainData.get(0).numAttributes() - 1) / 2];
		Map<Double, Double> tempAttributes = new HashMap<>();
		Instances tempData = new Instances(trainData);
		tempData.removeAll(tempData);

		for (int i = 0; i < trainData.numInstances(); i++) {
			for (int j = 0; j < (trainData.numAttributes() - 1) / 2; j++) {
				tempDistances[j] = trainData.get(i).value(2 * j);
				tempAttributes.put(trainData.get(i).value(2 * j), trainData.get(i).value(2 * j + 1));
			}

			Arrays.sort(tempDistances);

			List<Double> tempValues = new ArrayList<Double>();

			for (double distance : tempDistances) {
				tempValues.add(distance);
				tempValues.add(tempAttributes.get(distance));
			}
			tempValues.add(trainData.get(i).classValue());// the class value.

			Instance tempInstance = new MyDenseInstance(1.0,
					tempValues.stream().mapToDouble(Double::doubleValue).toArray());

			tempInstance.setDataset(trainData);
			tempData.add(tempInstance);
		}

		return tempData;
	}

	public double predictCoE(double[][] layout) {

		try {
			return classifier.classifyInstance(toInstance_Raw(layout));
		} catch (Exception e) {
			e.printStackTrace();
			return Double.MAX_VALUE;
		}
	}

	public void saveARFF(Instances dataSet, String filePath) {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);
	}

	public double run() {

		initialize();
		evaluateES();

		do {
			breeding();
			evaluateES();
		} while (WindFarmLayoutEvaluator.getNumberOfEvaluation() < trainEvaluations);

		System.out.println("Training Evaluations:" + WindFarmLayoutEvaluator.getNumberOfEvaluation());
		System.out.println("Colected Data Size:" + layoutsData.size());

		// Using collected data to train a surrogate model, then use it as a predictor.
		trainClassifier();

		do {
			breeding();
			evaluate_ML();
		} while (WindFarmLayoutEvaluator.getNumberOfEvaluation() < maxEvaluations);

		System.out.println("Searching Evaluations:" + WindFarmLayoutEvaluator.getNumberOfEvaluation());

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

	public String getDataFormat() {
		return dataFormat;
	}

	public void setDataFormat(String dataFormat) {
		this.dataFormat = dataFormat;
	}

	public int getMaxEvaluations() {
		return maxEvaluations;
	}

	public void setMaxEvaluations(int maxEvaluations) {
		this.maxEvaluations = maxEvaluations + WindFarmLayoutEvaluator.getNumberOfEvaluation();
	}

	public void setTrainEvaluations(int trainEvaluations) {
		this.trainEvaluations = trainEvaluations + WindFarmLayoutEvaluator.getNumberOfEvaluation();
	}

	public String getOperatorFlag() {
		return operatorFlag;
	}

	public void setOperatorFlag(String operatorFlag) {
		this.operatorFlag = operatorFlag;
	}

	public Classifier getClassifier() {
		return classifier;
	}

	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
	}

	public HashMap<double[][], Double> getLayoutsData() {
		return layoutsData;
	}

	public Instances getTrainData() {
		return trainData;
	}

}
