package main;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * MuLambdaES employs Mu-Lambda Evolutionary Strategy to optimize wind farm layout for certain given wind
 * scenarios. In short, Mu is the number of parents which survive, and Lambda is the number of kids that the μ
 * parents make in total. Notice that Lambda should be a multiple of Mu. ES practitioners usually refer to
 * their algorithm by the choice of Mu & Lambda. For example, if Mu = 5 and λ = 20, then we have a (5, 20) -
 * ES.
 * 
 * @author zhengchen
 * @version Best stratergy
 */

public class MuLambdaESML_Best extends MuLambdaESML {

	public MuLambdaESML_Best(WindFarmLayoutEvaluator evaluator) {
		super(evaluator);
	}

	/*
	 * In the best strategy, the lambda_star is defined as 1/2 of lambda.
	 */
	@Override
	public void setLambda(int lambda) {
		super.setLambda(lambda);
		lambda_star = lambda / 2;
	}

	/*
	 * In the (mu, lambda) - Best strategy, all lambda offspring are first evaluated using the surrogate.
	 * Then, lambda_star <= lambda best individuals according to the surrogate are re-evaluated using the
	 * expensive real fitness fuction.
	 */
	@Override
	public void evaluate_ML() {

		// Using the surrogate model to evaluate all lambda offspring.
		for (int p = 0; p < populations.size(); p++) {
			double[][] layout = populations.get(p);

			double coe_predicted;

			if (wfle.checkConstraint(layout)) {
				coe_predicted = predictCoE(layout);
			} else {
				coe_predicted = Double.MAX_VALUE;
			}

			fitnesses[p] = coe_predicted;
		}

		// Using the surrogate model to evaluate all the lambda offsprings to find out the best lambda_star
		// individuals.
		ArrayList<Double> tempFitnesses = new ArrayList<>();
		for (Double fitness : fitnesses) {
			tempFitnesses.add(fitness);
		}

		Arrays.sort(fitnesses);

		// Using the expensive real evaluation function to re-evaluate these lambda_star individuals.
		ArrayList<Double> temp_fitnesses = new ArrayList<>();
		for (int i = 0; i < lambda_star; i++) {
			wfle.evaluate(populations.get(tempFitnesses.indexOf(fitnesses[i])));
			fitnesses[i] = wfle.getEnergyCost();
		}

		// Finding out is there any improvement.
		double minFitness = Double.MAX_VALUE;
		for (int p = 0; p < fitnesses.length; p++) {
			if (fitnesses[p] < minFitness) {
				minFitness = fitnesses[p];
			}

			if (minFitness < bestFitness) {
				bestFitness = minFitness;
				System.out.println(bestFitness);
			}
		}

	}

	public void breeding_ML() {

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

		ArrayList<double[][]> temp = populations;
		populations = new ArrayList<>();

		// Add the winner back into the population, so a better solution might survive lots of generations.
		// for (int index : winners) {
		// populations.add(temp.get(index));
		// }

		// Generator new individuals using winners as parents
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

	@Override
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
		trainClassifier(classifier);

		do {
			breeding_ML();
			evaluate_ML();
		} while (WindFarmLayoutEvaluator.getNumberOfEvaluation() < maxEvaluations);

		System.out.println("Searching Evaluations:" + WindFarmLayoutEvaluator.getNumberOfEvaluation());

		return bestFitness;

	}
}
