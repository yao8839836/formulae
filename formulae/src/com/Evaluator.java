package com;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Evaluator {
	/*
	 * labeled LDA ten fold
	 */
	public static void ten_fold(int[][] med_doc, int K, int V,
			String[] med_doc_keys, Map<String, Set<Integer>> label_map,
			int[] labels) {

		double sum = 0;
		for (int i = 0; i < 10; i++) {

			if (i < 9) {

				double[][] alpha = new double[2509][K];
				int[][] train_set = new int[2509][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				labeledLDA l = new labeledLDA(train_set, V);

				int iterations = 1000;

				l.markovChain(K, alpha, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						predict_result = l.predict(med_doc[j], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 280);
				sum += (double) correct_count / 280;
			} else {

				double[][] alpha = new double[2520][K];
				int[][] train_set = new int[2520][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				labeledLDA l = new labeledLDA(train_set, V);

				int iterations = 1000;

				l.markovChain(K, alpha, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						predict_result = l.predict(med_doc[j], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 269);
				sum += (double) correct_count / 269;
			}

		}
		System.out.println(sum + "\n");

	}

	/*
	 * Labeled-LDA multi-label ten fold
	 */

	public static void ten_fold(int[][] med_doc, int K, int V,
			List<List<Integer>> docs_labels) {

		double sum = 0;

		double average_precision = 0, average_recall = 0, average_f_measure = 0;
		for (int i = 0; i < 5; i++) {

			double[][] alpha = new double[2472][K];
			int[][] train_set = new int[2472][K];
			int new_doc_number = 0;
			for (int j = 0; j < med_doc.length; j++) {
				if (j < (i + 1) * 618 && j >= i * 618) {
					continue;
				} else {
					train_set[new_doc_number] = med_doc[j];

					for (int e : docs_labels.get(j))

						alpha[new_doc_number][e] = (double) 50 / K;

					new_doc_number++;
				}
			}
			double beta = 0.1;
			labeledLDA l = new labeledLDA(train_set, V);

			int iterations = 1000;

			l.markovChain(K, alpha, beta, iterations);
			// double [][] theta = l.estimateTheta();
			double[][] phi = l.estimatePhi();

			int correct_count = 0;
			int precision_count = 0, total_prediction = 0, total_real = 0;

			double[] predict_result = new double[K];// predict
			for (int j = 0; j < med_doc.length; j++) {

				if (j < (i + 1) * 618 && j >= i * 618) {

					predict_result = l.predict(med_doc[j], phi);
					int predict_label = Common.maxIndex(predict_result);
					List<Integer> real = docs_labels.get(j);
					if (real.contains(predict_label))
						correct_count++;

					double[] normalized_result = Common
							.normalize(predict_result);

					Set<Integer> predict_set = new HashSet<Integer>();

					for (int n = 0; n < 3; n++) {
						predict_label = Common.maxIndex(normalized_result);

						predict_set.add(predict_label);
						normalized_result[predict_label] = 0;
					}

					total_prediction += predict_set.size();

					total_real += real.size();

					for (int e : predict_set) {
						if (real.contains(e)) {
							precision_count++;
						}
					}
				}

			}
			System.out.println((double) correct_count / 618);
			sum += (double) correct_count / 618;

			double precision = (double) precision_count / total_prediction;
			double recall = (double) precision_count / total_real;
			double f_measure = 2 * precision * recall / (precision + recall);

			average_precision += precision;
			average_recall += recall;
			average_f_measure += f_measure;

			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("f_measure : " + f_measure);
		}

		System.out.println("labeled-LDA ten fold " + sum / 5 + "\n");

		System.out.println("average-precision : " + average_precision / 5);
		System.out.println("average-recall : " + average_recall / 5);
		System.out.println("average-f_measure : " + average_f_measure / 5);

	}

	/*
	 * labeled-LDA leave one
	 */
	public static void leave_one(int[][] med_doc, int[] labels, int K, int V) {

		int correct_count = 0;
		for (int i = 0, D = med_doc.length; i < D; i++) {

			double[][] alpha = new double[D - 1][K];
			int[][] train_set = new int[D - 1][K];

			int new_doc_number = 0;

			for (int j = 0; j < med_doc.length; j++) {
				if (j == i) {
					continue;
				} else {
					train_set[new_doc_number] = med_doc[j];
					alpha[new_doc_number][labels[j]] = (double) 50 / K;
					new_doc_number++;
				}
			}

			double beta = 0.1;
			labeledLDA l = new labeledLDA(train_set, V);

			int iterations = 1000;

			l.markovChain(K, alpha, beta, iterations);
			// double [][] theta = l.estimateTheta();
			double[][] phi = l.estimatePhi();

			double[] predict_result = new double[K];// predict
			for (int j = 0; j < med_doc.length; j++) {

				if (j == i) {

					predict_result = l.predict(med_doc[j], phi);
					int predict_label = Common.maxIndex(predict_result);
					if (predict_label == labels[j])
						correct_count++;

				}
			}

		}
		System.out.println("labeled-LDA leave one : " + (double) correct_count
				/ med_doc.length);

	}

	/*
	 * zlabel-labeledLDA ten fold
	 */
	public static void ten_fold(int[][] med_doc, int K, int V,
			Map<Integer, Set<Integer>> zset, String[] med_doc_keys,
			Map<String, Set<Integer>> label_map, int[] labels) {

		double sum = 0;
		for (int i = 0; i < 10; i++) {

			if (i < 9) {

				double[][] alpha = new double[2509][K];
				int[][] train_set = new int[2509][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				ZsetLabeledLDA l = new ZsetLabeledLDA(train_set, V, zset);

				int iterations = 1000;

				l.markovChain(K, alpha, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						predict_result = l.predict(med_doc[j], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 280);
				sum += (double) correct_count / 280;
			} else {

				double[][] alpha = new double[2520][K];
				int[][] train_set = new int[2520][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				ZsetLabeledLDA l = new ZsetLabeledLDA(train_set, V, zset);

				int iterations = 1000;

				l.markovChain(K, alpha, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						predict_result = l.predict(med_doc[j], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 269);
				sum += (double) correct_count / 269;
			}

		}
		System.out.println(sum + "\n");

	}

	/*
	 * zset-Labeled-LDA multi-label ten fold
	 */

	public static void ten_fold(int[][] med_doc, int K, int V,
			List<List<Integer>> docs_labels, Map<Integer, Set<Integer>> zset) {

		double sum = 0;

		double average_precision = 0, average_recall = 0, average_f_measure = 0;
		for (int i = 0; i < 5; i++) {

			double[][] alpha = new double[2472][K];
			int[][] train_set = new int[2472][K];
			int new_doc_number = 0;
			for (int j = 0; j < med_doc.length; j++) {
				if (j < (i + 1) * 618 && j >= i * 618) {
					continue;
				} else {
					train_set[new_doc_number] = med_doc[j];

					for (int e : docs_labels.get(j))

						alpha[new_doc_number][e] = (double) 50 / K;

					new_doc_number++;
				}
			}
			double beta = 0.1;
			ZsetLabeledLDA l = new ZsetLabeledLDA(train_set, V, zset);

			int iterations = 1000;

			l.markovChain(K, alpha, beta, iterations);
			// double [][] theta = l.estimateTheta();
			double[][] phi = l.estimatePhi();

			int correct_count = 0;
			int precision_count = 0, total_prediction = 0, total_real = 0;

			double[] predict_result = new double[K];// predict
			for (int j = 0; j < med_doc.length; j++) {

				if (j < (i + 1) * 618 && j >= i * 618) {

					predict_result = l.predict(med_doc[j], phi);
					int predict_label = Common.maxIndex(predict_result);
					List<Integer> real = docs_labels.get(j);
					if (real.contains(predict_label))
						correct_count++;

					double[] normalized_result = Common
							.normalize(predict_result);

					Set<Integer> predict_set = new HashSet<Integer>();

					for (int n = 0; n < 3; n++) {
						predict_label = Common.maxIndex(normalized_result);

						predict_set.add(predict_label);
						normalized_result[predict_label] = 0;
					}

					total_prediction += predict_set.size();

					total_real += real.size();

					for (int e : predict_set) {
						if (real.contains(e)) {
							precision_count++;
						}
					}
				}

			}
			System.out.println((double) correct_count / 618);
			sum += (double) correct_count / 618;

			double precision = (double) precision_count / total_prediction;
			double recall = (double) precision_count / total_real;
			double f_measure = 2 * precision * recall / (precision + recall);

			average_precision += precision;
			average_recall += recall;
			average_f_measure += f_measure;

			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("f_measure : " + f_measure);
		}

		System.out.println("zset-labeled-LDA ten fold " + sum / 5 + "\n");

		System.out.println("average-precision : " + average_precision / 5);
		System.out.println("average-recall : " + average_recall / 5);
		System.out.println("average-f_measure : " + average_f_measure / 5);

	}

	/*
	 * zlabel-labeled-LDA leave one
	 */
	public static void leave_one(int[][] med_doc, int[] labels, int K, int V,
			Map<Integer, Set<Integer>> zset) {

		int correct_count = 0;
		for (int i = 0, D = med_doc.length; i < D; i++) {

			double[][] alpha = new double[D - 1][K];
			int[][] train_set = new int[D - 1][K];

			int new_doc_number = 0;

			for (int j = 0; j < med_doc.length; j++) {
				if (j == i) {
					continue;
				} else {
					train_set[new_doc_number] = med_doc[j];
					alpha[new_doc_number][labels[j]] = (double) 50 / K;
					new_doc_number++;
				}
			}

			double beta = 0.1;
			ZsetLabeledLDA l = new ZsetLabeledLDA(train_set, V, zset);

			int iterations = 1000;

			l.markovChain(K, alpha, beta, iterations);
			// double [][] theta = l.estimateTheta();
			double[][] phi = l.estimatePhi();

			double[] predict_result = new double[K];// predict
			for (int j = 0; j < med_doc.length; j++) {

				if (j == i) {

					predict_result = l.predict(med_doc[j], phi);
					int predict_label = Common.maxIndex(predict_result);
					if (predict_label == labels[j])
						correct_count++;

				}
			}

		}
		System.out.println("zlabel-labeled-LDA leave one : "
				+ (double) correct_count / med_doc.length);

	}

	/*
	 * Formula-LDA ten_fold
	 */
	public static void ten_fold(int[][] med_doc, int K, int V,
			Map<Integer, Set<Integer>> zset,
			Map<Integer, Set<Integer>> z_cannot_set,
			Map<Integer, MedConstraint> constraint, String[] med_doc_keys,
			Map<String, Set<Integer>> label_map, int[] labels) {

		double sum = 0;
		for (int i = 0; i < 10; i++) {

			if (i < 9) {

				double[][] alpha = new double[2509][K];
				int[][] train_set = new int[2509][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				FormulaLDA l = new FormulaLDA(train_set, V, zset, z_cannot_set,
						constraint);

				int iterations = 1000;

				l.markovChain(K, alpha, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						predict_result = l.predict(med_doc[j], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 280);
				sum += (double) correct_count / 280;
			} else {

				double[][] alpha = new double[2520][K];
				int[][] train_set = new int[2520][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				FormulaLDA l = new FormulaLDA(train_set, V, zset, z_cannot_set,
						constraint);

				int iterations = 1000;

				l.markovChain(K, alpha, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						predict_result = l.predict(med_doc[j], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 269);
				sum += (double) correct_count / 269;
			}

		}
		System.out.println(sum + "\n");

	}

	/*
	 * Formula-LDA multi-label ten fold
	 */

	public static void ten_fold(int[][] med_doc, int K, int V,
			List<List<Integer>> docs_labels, Map<Integer, Set<Integer>> zset,
			Map<Integer, Set<Integer>> z_cannot_set,
			Map<Integer, MedConstraint> constraint) {

		double sum = 0;

		double average_precision = 0, average_recall = 0, average_f_measure = 0;
		for (int i = 0; i < 5; i++) {

			double[][] alpha = new double[2472][K];
			int[][] train_set = new int[2472][K];
			int new_doc_number = 0;
			for (int j = 0; j < med_doc.length; j++) {
				if (j < (i + 1) * 618 && j >= i * 618) {
					continue;
				} else {
					train_set[new_doc_number] = med_doc[j];

					for (int e : docs_labels.get(j))

						alpha[new_doc_number][e] = (double) 50 / K;

					new_doc_number++;
				}
			}
			double beta = 0.1;
			FormulaLDA l = new FormulaLDA(train_set, V, zset, z_cannot_set,
					constraint);

			int iterations = 1000;

			l.markovChain(K, alpha, beta, iterations);
			// double [][] theta = l.estimateTheta();
			double[][] phi = l.estimatePhi();

			int correct_count = 0;
			int precision_count = 0, total_prediction = 0, total_real = 0;

			double[] predict_result = new double[K];// predict
			for (int j = 0; j < med_doc.length; j++) {

				if (j < (i + 1) * 618 && j >= i * 618) {

					predict_result = l.predict(med_doc[j], phi);
					int predict_label = Common.maxIndex(predict_result);
					List<Integer> real = docs_labels.get(j);
					if (real.contains(predict_label))
						correct_count++;

					double[] normalized_result = Common
							.normalize(predict_result);

					Set<Integer> predict_set = new HashSet<Integer>();

					// for (int k = 0; k < normalized_result.length; k++) {
					// if (normalized_result[k] >= (double) 0.5 / K)
					// predict_set.add(k);
					// }

					for (int n = 0; n < 3; n++) {
						predict_label = Common.maxIndex(normalized_result);

						predict_set.add(predict_label);
						normalized_result[predict_label] = 0;
					}

					total_prediction += predict_set.size();

					total_real += real.size();

					for (int e : predict_set) {
						if (real.contains(e)) {
							precision_count++;
						}
					}
				}

			}
			System.out.println((double) correct_count / 618);
			sum += (double) correct_count / 618;

			double precision = (double) precision_count / total_prediction;
			double recall = (double) precision_count / total_real;
			double f_measure = 2 * precision * recall / (precision + recall);
			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("f_measure : " + f_measure);

			average_precision += precision;
			average_recall += recall;
			average_f_measure += f_measure;
		}

		System.out.println("Formula-LDA ten fold " + sum / 5 + "\n");

		System.out.println("average-precision : " + average_precision / 5);
		System.out.println("average-recall : " + average_recall / 5);
		System.out.println("average-f_measure : " + average_f_measure / 5);

	}

	/*
	 * labeled-LDA leave one
	 */
	public static void leave_one(int[][] med_doc, int[] labels, int K, int V,
			Map<Integer, Set<Integer>> zset,
			Map<Integer, Set<Integer>> z_cannot_set,
			Map<Integer, MedConstraint> constraint) {

		int correct_count = 0;
		for (int i = 0, D = med_doc.length; i < D; i++) {

			double[][] alpha = new double[D - 1][K];
			int[][] train_set = new int[D - 1][K];

			int new_doc_number = 0;

			for (int j = 0; j < med_doc.length; j++) {
				if (j == i) {
					continue;
				} else {
					train_set[new_doc_number] = med_doc[j];
					alpha[new_doc_number][labels[j]] = (double) 50 / K;
					new_doc_number++;
				}
			}

			double beta = 0.1;
			FormulaLDA l = new FormulaLDA(train_set, V, zset, z_cannot_set,
					constraint);

			int iterations = 1000;

			l.markovChain(K, alpha, beta, iterations);
			// double [][] theta = l.estimateTheta();
			double[][] phi = l.estimatePhi();

			double[] predict_result = new double[K];// predict
			for (int j = 0; j < med_doc.length; j++) {

				if (j == i) {

					predict_result = l.predict(med_doc[j], phi);
					int predict_label = Common.maxIndex(predict_result);
					if (predict_label == labels[j])
						correct_count++;

				}
			}

		}
		System.out.println("Formula-LDA leave one : " + (double) correct_count
				/ med_doc.length);

	}

	/*
	 * Naive Bayes ten fold
	 */
	public static void ten_fold(Instances ins, Corpus c, int[] order)
			throws Exception {

		Instances train_instances = new Instances(ins);

		double sum = 0;
		for (int i = 0; i < 10; i++) {

			train_instances.delete();

			for (int j = 0, total_length = ins.numInstances(); j < total_length; j++) {

				if (j < (i + 1) * 280 && j >= i * 280) {
					continue;
				} else {
					train_instances.add(ins.instance(j));
				}
			}

			NaiveBayes n = new NaiveBayes();

			n.buildClassifier(train_instances);

			int correct = 0;

			for (int j = 0, total_length = ins.numInstances(); j < total_length; j++) {

				if (j < (i + 1) * 280 && j >= i * 280) {

					Instance testInst = ins.instance(j);

					// Set<Integer> label_set = c.label_map
					// .get(c.map_keys[order[j]]);

					if (testInst.classValue() == n.classifyInstance(testInst))
						correct++;

					// if (label_set.contains((int)
					// n.classifyInstance(testInst)))
					// correct++;
				}
			}
			if (i != 9) {
				double correct_rate = (double) correct / 280;
				System.out.println(correct_rate);
				sum += correct_rate;
			} else {
				double correct_rate = (double) correct / 269;
				System.out.println(correct_rate);
				sum += correct_rate;
			}
		}
		System.out.println(sum + "\n");
	}

	/*
	 * Naive Bayes leave one
	 */
	public static void leave_one(Instances ins) throws Exception {

		Instances train_instances = new Instances(ins);

		int correct = 0;
		for (int i = 0; i < ins.numInstances(); i++) {

			train_instances.delete();

			for (int j = 0, total_length = ins.numInstances(); j < total_length; j++) {

				if (j == i) {
					continue;
				} else {
					train_instances.add(ins.instance(j));
				}
			}

			NaiveBayes n = new NaiveBayes();

			n.buildClassifier(train_instances);

			for (int j = 0, total_length = ins.numInstances(); j < total_length; j++) {

				if (j == i) {

					Instance testInst = ins.instance(j);

					if (testInst.classValue() == n.classifyInstance(testInst))
						correct++;
				}
			}

		}
		System.out.println("Naive Bayes leave one : " + correct + "\n");
	}

	public static void btm_ten_fold(int[][] med_doc, int K, int V, int[] labels) {

		double sum = 0;
		for (int i = 0; i < 10; i++) {

			if (i < 9) {

				double[][] alpha = new double[2509][K];
				int[][] train_set = new int[2509][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				BTM l = new BTM(med_doc, V, labels);

				int iterations = 1000;

				l.markovChain(K, beta, iterations);
				// double [][] theta = l.estimateTheta();
				double[][] phi = l.estimatePhi();

				// double[] btm_theta = l.estimateTheta();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						// predict_result = l.doc_infer_sum_w(med_doc[j], phi,
						// btm_theta);

						predict_result = l.predict(med_doc[i], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 280);
				sum += (double) correct_count / 280;
			} else {

				double[][] alpha = new double[2520][K];
				int[][] train_set = new int[2520][K];
				int new_doc_number = 0;
				for (int j = 0; j < med_doc.length; j++) {
					if (j < (i + 1) * 280 && j >= i * 280) {
						continue;
					} else {
						train_set[new_doc_number] = med_doc[j];

						alpha[new_doc_number][labels[j]] = (double) 50 / K;

						new_doc_number++;
					}
				}
				double beta = 0.1;
				BTM l = new BTM(med_doc, V, labels);

				int iterations = 1000;

				l.markovChain(K, beta, iterations);
				// double [][] theta = l.estimateTheta();

				double[][] phi = l.estimatePhi();

				// double[] btm_theta = l.estimateTheta();

				int correct_count = 0;
				double[] predict_result = new double[K];// predict
				for (int j = 0; j < med_doc.length; j++) {

					if (j < (i + 1) * 280 && j >= i * 280) {

						// predict_result = l.doc_infer_sum_w(med_doc[j], phi,
						// btm_theta);
						predict_result = l.predict(med_doc[i], phi);
						int predict_label = Common.maxIndex(predict_result);

						if (predict_label == labels[j])
							correct_count++;
					}

				}
				System.out.println((double) correct_count / 269);
				sum += (double) correct_count / 269;
			}

		}
		System.out.println(sum + "\n");
	}

	/*
	 * SVM five fold
	 */
	public static void ten_fold(List<Set<Integer>> real_docs_labels)
			throws Exception {

		File result = new File("src//file//svm_result.txt");
		OutputStream out = new FileOutputStream(result, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out,
				"utf-8"));

		List<Instances> all_set = new ArrayList<Instances>();// 每个分类器的所有数据

		for (int k = 0; k < 28; k++) {

			File file = new File("src//file//svm//topic" + k + ".arff");

			ArffLoader loader = new ArffLoader();

			loader.setFile(file);

			Instances ins_all = loader.getDataSet();
			ins_all.setClassIndex(ins_all.numAttributes() - 1);

			all_set.add(ins_all);

		}

		double average_precision = 0, average_recall = 0, average_f_measure = 0;

		for (int i = 0; i < 5; i++) {

			for (int k = 0; k < 28; k++) {

				Instances all_instances_k = all_set.get(k);
				Instances train_instances = new Instances(all_instances_k);

				train_instances.delete();

				for (int j = 0, total_length = all_instances_k.numInstances(); j < total_length; j++) {

					if (j % 5 == i) { // if (j < (i + 1) * 618 && j >= i * 618)
						continue;
					} else {
						train_instances.add(all_instances_k.instance(j));
					}
				}

				train_instances
						.setClassIndex(all_instances_k.numAttributes() - 1);

				LibSVM libsvm = new LibSVM();

				libsvm.setCost(100);

				System.out.println("Start training....." + k);

				libsvm.buildClassifier(train_instances);

				System.out.println("build complete " + k);

				Common.SaveModel(libsvm, "svm" + k + ".model");
			}

			/*
			 * 读取训练好的分类器
			 */
			List<LibSVM> svms = new ArrayList<LibSVM>();

			for (int k = 0; k < 28; k++) {
				LibSVM libsvm = (LibSVM) Common
						.LoadModel("src//file//svm//model//svm" + k + ".model");
				svms.add(libsvm);
			}

			/*
			 * 测试剩下的10%
			 */
			int precision_count = 0, total_prediction = 0, total_real = 0;

			for (int j = 0, total_length = all_set.get(0).numInstances(); j < total_length; j++) {

				if (j % 5 == i) {

					Set<Integer> predict = new HashSet<Integer>();

					Set<Integer> real = real_docs_labels.get(j);

					for (int k = 0; k < 28; k++) {

						Instance testInst = all_set.get(k).instance(j);

						LibSVM libsvm = svms.get(k);
						int predict_result = (int) libsvm
								.classifyInstance(testInst);
						if (predict_result == 1)
							predict.add(k);
						System.out.print(predict_result + " ");
					}

					total_prediction += predict.size();

					total_real += real.size();

					for (int e : predict) {
						if (real.contains(e))
							precision_count++;
					}

				}

			}
			double precision = (double) precision_count / total_prediction;
			double recall = (double) precision_count / total_real;
			double f_measure = 2 * precision * recall / (precision + recall);

			average_precision += precision;
			average_recall += recall;
			average_f_measure += f_measure;

			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("f_measure : " + f_measure);

			bw.write("precision : " + precision + "\n");
			bw.write("recall : " + recall + "\n");
			bw.write("f_measure : " + f_measure + "\n");

		}
		System.out.println("average-precision : " + average_precision / 5);
		System.out.println("average-recall : " + average_recall / 5);
		System.out.println("average-f_measure : " + average_f_measure / 5);

		bw.write("average-precision : " + average_precision / 5 + "\n");
		bw.write("average-recall : " + average_recall / 5 + "\n");
		bw.write("average-f_measure : " + average_f_measure / 5 + "\n");

		bw.close();
		out.close();

	}

	/*
	 * Multi Logistic five fold
	 */
	public static void logistic_five_fold(List<Set<Integer>> real_docs_labels)
			throws Exception {

		File result = new File("src//file//logistic_result.txt");
		OutputStream out = new FileOutputStream(result, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out,
				"utf-8"));

		List<Instances> all_set = new ArrayList<Instances>();// 每个分类器的所有数据

		for (int k = 0; k < 28; k++) {

			File file = new File("src//file//svm//topic" + k + ".arff");

			ArffLoader loader = new ArffLoader();

			loader.setFile(file);

			Instances ins_all = loader.getDataSet();
			ins_all.setClassIndex(ins_all.numAttributes() - 1);

			all_set.add(ins_all);

		}

		double average_precision = 0, average_recall = 0, average_f_measure = 0;

		for (int i = 0; i < 5; i++) {

			for (int k = 0; k < 28; k++) {

				Instances all_instances_k = all_set.get(k);
				Instances train_instances = new Instances(all_instances_k);

				train_instances.delete();

				for (int j = 0, total_length = all_instances_k.numInstances(); j < total_length; j++) {

					if (j % 5 == i) { // if (j < (i + 1) * 618 && j >= i * 618)
						continue;
					} else {
						train_instances.add(all_instances_k.instance(j));
					}
				}

				train_instances
						.setClassIndex(all_instances_k.numAttributes() - 1);

				Logistic n = new Logistic();

				System.out.println("Start training....." + k);

				n.buildClassifier(train_instances);

				System.out.println("build complete " + k);

				Common.SaveModel(n, "logistic" + k + ".model");
			}

			/*
			 * 读取训练好的分类器
			 */
			List<Logistic> logistics = new ArrayList<Logistic>();

			for (int k = 0; k < 28; k++) {
				Logistic n = (Logistic) Common
						.LoadModel("src//file//svm//model//logistic" + k
								+ ".model");
				logistics.add(n);
			}

			/*
			 * 测试剩下的10%
			 */
			int precision_count = 0, total_prediction = 0, total_real = 0;

			for (int j = 0, total_length = all_set.get(0).numInstances(); j < total_length; j++) {

				if (j % 5 == i) { // if (j < (i + 1) * 618 && j >= i * 618)

					Set<Integer> predict = new HashSet<Integer>();

					Set<Integer> real = real_docs_labels.get(j);

					for (int k = 0; k < 28; k++) {

						Instance testInst = all_set.get(k).instance(j);

						Logistic n = logistics.get(k);
						int predict_result = (int) n.classifyInstance(testInst);
						if (predict_result == 1)
							predict.add(k);
						System.out.print(predict_result + " ");
					}

					total_prediction += predict.size();

					total_real += real.size();

					for (int e : predict) {
						if (real.contains(e))
							precision_count++;
					}

				}

			}
			double precision = (double) precision_count / total_prediction;
			double recall = (double) precision_count / total_real;
			double f_measure = 2 * precision * recall / (precision + recall);

			average_precision += precision;
			average_recall += recall;
			average_f_measure += f_measure;

			System.out.println("precision : " + precision);
			System.out.println("recall : " + recall);
			System.out.println("f_measure : " + f_measure);

			bw.write("precision : " + precision + "\n");
			bw.write("recall : " + recall + "\n");
			bw.write("f_measure : " + f_measure + "\n");

		}
		System.out.println("average-precision : " + average_precision / 5);
		System.out.println("average-recall : " + average_recall / 5);
		System.out.println("average-f_measure : " + average_f_measure / 5);

		bw.write("average-precision : " + average_precision / 5 + "\n");
		bw.write("average-recall : " + average_recall / 5 + "\n");
		bw.write("average-f_measure : " + average_f_measure / 5 + "\n");

		bw.close();
		out.close();

	}

}
