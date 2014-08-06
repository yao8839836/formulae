package com;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class PredictOnMultiLabelClinicalCase {

	public static void main(String[] args) throws IOException, SQLException,
			InterruptedException {

		MultiLabelClinicalCases mcc = Preprocess.getMultiLabelClinical();

		int[][] med_doc = mcc.med_doc;

		List<List<String>> doc_labels = mcc.multilabels;

		List<String> label_set = mcc.label_set;

		int K = label_set.size();

		List<String> vocab = mcc.vocab;

		int V = vocab.size();

		System.out.println(V);

		List<List<Integer>> doc_labels_int = new ArrayList<List<Integer>>();

		double[][] alpha = new double[med_doc.length][K];

		for (int i = 0; i < med_doc.length; i++) {

			List<String> doc_label = doc_labels.get(i);

			List<Integer> doc_label_int = new ArrayList<Integer>();

			for (String label : doc_label) {

				int label_index = label_set.indexOf(label);
				alpha[i][label_index] = (double) 50 / K;

				doc_label_int.add(label_index);

			}
			doc_labels_int.add(doc_label_int);

		}

		double beta = 0.1;

		/*
		 * zset
		 */

		MedLabels m = Preprocess.getZLabels();

		Map<Integer, Set<Integer>> zset = new HashMap<Integer, Set<Integer>>();

		for (int i = 0; i < V; i++) {

			Set<Integer> zlabel = new HashSet<Integer>();
			for (int j = 0, length = m.medicines.length; j < length; j++) {

				String medicine = m.medicines[j];

				String med_class = m.med_classes[j];

				for (int k = 0; k < K; k++) {

					if (vocab.get(i).equals(medicine)) {

						if (label_set.get(k).contains(
								med_class.substring(0, med_class.indexOf("药")))) {
							zlabel.add(k);
						}
					}
				}

			}
			zset.put(i, zlabel);
			System.out.println(i + "," + zlabel.toString());
		}

		/*
		 * 相畏相反，相辅相成
		 */

		List<String> constraints = Preprocess.getMustCannotLink();

		float[][] matrix = new float[V][V];

		for (String e : constraints) {
			// System.out.println(e);
			String[] tuple = e.split(" ");

			for (int v = 0; v < V; v++) {
				if (tuple[0].equals(vocab.get(v))) {

					if (vocab.contains(tuple[1])) {

						if (tuple[2].equals("相辅相成")) {
							matrix[v][vocab.indexOf(tuple[1])] = 1;
							matrix[vocab.indexOf(tuple[1])][v] = 1;
						} else {
							matrix[v][vocab.indexOf(tuple[1])] = (float) 0.5;
							matrix[vocab.indexOf(tuple[1])][v] = (float) 0.5;
						}
					}
				} else if (tuple[1].equals(vocab.get(v))) {

					if (vocab.contains(tuple[0])) {

						if (tuple[2].equals("相辅相成")) {
							matrix[vocab.indexOf(tuple[0])][v] = 1;
							matrix[vocab.indexOf(tuple[0])][v] = 1;
						} else {
							matrix[v][vocab.indexOf(tuple[0])] = (float) 0.5;
							matrix[vocab.indexOf(tuple[0])][v] = (float) 0.5;
						}
					}
				}
			}

		}

		Map<Integer, MedConstraint> constrain_map = new HashMap<Integer, MedConstraint>();

		for (int i = 0; i < V; i++) {

			Set<Integer> must_link = new HashSet<Integer>();
			Set<Integer> cannot_link = new HashSet<Integer>();

			for (int j = 0; j < V; j++) {
				if (matrix[i][j] != 0) {

					if (matrix[i][j] == 1) {

						must_link.add(j);
						// j增加 药 label
						if (zset.get(j).isEmpty() && !zset.get(i).isEmpty()) {

							// 通过相辅相成增加与某位药i相辅相成的其他药的主题范围（如果没有）?
							zset.put(j, zset.get(i));
							// System.out.println(j + " " + zset.get(i));
						} else if (!zset.get(j).isEmpty()
								&& zset.get(i).isEmpty()) {
							zset.put(i, zset.get(j));
							// System.out.println(i + " " +zset.get(j));
						}
					} else {

						cannot_link.add(j);

					}

				}
			}

			MedConstraint med_constraint = new MedConstraint(must_link,
					cannot_link);

			constrain_map.put(i, med_constraint);

			// System.out.println(i + ":" + constrain_map.get(i).must_link + ","
			// + constrain_map.get(i).cannot_link);

		}

		Map<Integer, Set<Integer>> z_cannot_set = new HashMap<Integer, Set<Integer>>();

		// 全部初始化

		for (int i = 0; i < V; i++)
			z_cannot_set.put(i, new HashSet<Integer>());

		for (int i = 0; i < V; i++) {

			Set<Integer> cannot = new HashSet<Integer>();

			Set<Integer> must = zset.get(i);

			if (!must.isEmpty())
				for (int k = 0; k < K; k++)
					if (!must.contains(k))
						cannot.add(k);

			z_cannot_set.put(i, cannot);

		}

		/*
		 * labeledLDA begins
		 */
		labeledLDA l = new labeledLDA(med_doc, V);

		int iterations = 1000;

		l.markovChain(K, alpha, beta, iterations);

		double[][] phi = l.estimatePhi();

		// double[][] theta = l.estimateTheta();

		double[][] theta_estimate = new double[med_doc.length][K];

		int correct_count = 0;

		int precision_count = 0, total_prediction = 0, total_real = 0;

		for (int i = 0; i < med_doc.length; i++) {

			double[] predict_result = l.predict(med_doc[i], phi);// predict
			// by Bayes rule

			// predict by estimate theta
			// double[] predict_result = l.predict(med_doc[i]);

			int predict_label = Common.maxIndex(predict_result);

			List<Integer> real = doc_labels_int.get(i);
			if (real.contains(predict_label))
				correct_count++;

			// 设置阈值，预测标签集合

			double[] normalized_result = Common.normalize(predict_result);

			Set<Integer> predict_set = new HashSet<Integer>();

			for (int j = 0; j < normalized_result.length; j++) {
				// if(normalized_result [j] > (double)2/ K)
				// predict_set.add(j);

			}

			for (int n = 0; n < 3; n++) {
				predict_label = Common.maxIndex(normalized_result);

				predict_set.add(predict_label);
				normalized_result[predict_label] = 0;
			}

			theta_estimate[i] = Common.normalize(predict_result);

			total_prediction += predict_set.size();

			total_real += real.size();

			for (int e : predict_set) {
				if (real.contains(e)) {
					precision_count++;
				}
			}

		}
		// 将主题比例和medicine一起作为feature写文件
		MultiLabelSVM.write_file(med_doc, doc_labels_int, K, V, theta_estimate);

		System.out.println("labeled-LDA: " + (double) correct_count
				/ med_doc.length);

		double precision = (double) precision_count / total_prediction;
		double recall = (double) precision_count / total_real;
		double f_measure = 2 * precision * recall / (precision + recall);
		System.out.println("precision : " + precision);
		System.out.println("recall : " + recall);
		System.out.println("f_measure : " + f_measure);

		/*
		 * 主题下的high probability medicines
		 */
		Map<String, Set<String>> class_med_map = Preprocess.getLabelMedicines();

		for (String key : class_med_map.keySet()) {
			System.out.println(key + ": " + class_med_map.get(key));
		}

		int count = 0;

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + label_set.get(k) + " ");

			Set<String> medicines = class_med_map.get(label_set.get(k));

			for (int i = 0; i < 10; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");

				if (medicines.contains(vocab.get(index))) {

					count++;
					System.out.print(1);
				}

				phi[k][index] = 0;
			}
			System.out.println();
		}

		System.out.println("labeldLDA accuracy : " + (double) count / (10 * K));

		/*
		 * zset-labeled LDA begins
		 */

		ZsetLabeledLDA zll = new ZsetLabeledLDA(med_doc, V, zset);

		zll.markovChain(K, alpha, beta, iterations);

		phi = zll.estimatePhi();

		correct_count = 0;

		for (int i = 0; i < med_doc.length; i++) {

			double[] predict_result = zll.predict(med_doc[i], phi);// predict

			int predict_label = Common.maxIndex(predict_result);

			if (doc_labels_int.get(i).contains(predict_label))
				correct_count++;
		}
		System.out.println("zset-labeled-LDA: " + (double) correct_count
				/ med_doc.length);

		count = 0;

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + label_set.get(k));

			Set<String> medicines = class_med_map.get(label_set.get(k));

			for (int i = 0; i < 10; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");

				if (medicines.contains(vocab.get(index))) {
					count++;
					System.out.print(1);
				}

				phi[k][index] = 0;
			}
			System.out.println();
		}

		System.out.println("zset-labeled-LDA accuracy : " + (double) count
				/ (10 * K));
		/*
		 * Formula-LDA begins
		 */

		FormulaLDA fl = new FormulaLDA(med_doc, V, zset, z_cannot_set,
				constrain_map);

		fl.markovChain(K, alpha, beta, iterations);

		phi = fl.estimatePhi();

		correct_count = 0;

		for (int i = 0; i < med_doc.length; i++) {

			double[] predict_result = fl.predict(med_doc[i], phi);// predict

			int predict_label = Common.maxIndex(predict_result);

			if (doc_labels_int.get(i).contains(predict_label))
				correct_count++;
		}
		System.out.println("Formula-LDA: " + (double) correct_count
				/ med_doc.length);

		count = 0;

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + label_set.get(k));

			Set<String> medicines = class_med_map.get(label_set.get(k));

			for (int i = 0; i < 10; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");

				if (medicines.contains(vocab.get(index))) {
					System.out.print(1);
					count++;
				}

				phi[k][index] = 0;
			}
			System.out.println();
		}

		System.out.println("Formula-LDA accuracy : " + (double) count
				/ (10 * K));

//		MultiLabelSVM.write_file(med_doc, doc_labels_int, K, V, theta_estimate);

		// MultiLabelSVM.write_file(med_doc, doc_labels_int, K, V);

		Evaluator.ten_fold(med_doc, K, V, doc_labels_int);

		Evaluator.ten_fold(med_doc, K, V, doc_labels_int, zset);

		Evaluator.ten_fold(med_doc, K, V, doc_labels_int, zset, z_cannot_set,
				constrain_map);

	}

}
