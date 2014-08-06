package com;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class PredictOnClinicalCase {

	public static void main(String args[]) throws Exception {

		Corpus c = Preprocess.getCorpus();

		int[][] med_doc = c.distinct_med_doc; // 训练文档集,已去重

		String[] med_doc_keys = c.distinct_map_keys; // 文档 key,例如 四倍汤,赤石脂 石菖蒲 牡蛎
		// 人参 食盐

		Map<String, Set<Integer>> label_map = c.label_map;

		List<String> label_set = c.label_set;

		int K = label_set.size();

		List<String> vocab = c.vocab;
		int V = vocab.size();

		double[][] alpha = new double[med_doc.length][K];

		for (int i = 0; i < med_doc.length; i++) {

			for (int e : label_map.get(med_doc_keys[i])) {
				alpha[i][e] = (double) 50 / K;
			}

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
		 * 读测试数据集
		 */
		ClinicalCases cc = Preprocess.getClinical();

		int[][] test_docs = cc.med_doc;

		String[] test_labels = cc.labels;

		int[] test_labels_int = new int[test_labels.length];

		for (int i = 0; i < test_labels_int.length; i++) {
			test_labels_int[i] = label_set.indexOf(test_labels[i]);
		}

		/*
		 * labeledLDA begins
		 */
		labeledLDA l = new labeledLDA(med_doc, V);

		int iterations = 1000;

		l.markovChain(K, alpha, beta, iterations);

		double[][] phi = l.estimatePhi();

		int correct_count = 0;

		for (int i = 0; i < test_docs.length; i++) {

			double[] predict_result = l.predict(test_docs[i], phi);// predict

			int predict_label = Common.maxIndex(predict_result);

			if (predict_label == test_labels_int[i])
				correct_count++;
		}
		System.out.println("labeled-LDA: " + (double) correct_count
				/ test_docs.length);

		/*
		 * zset-labeled LDA begins
		 */

		ZsetLabeledLDA zll = new ZsetLabeledLDA(med_doc, V, zset);

		zll.markovChain(K, alpha, beta, iterations);

		phi = zll.estimatePhi();

		correct_count = 0;

		for (int i = 0; i < test_docs.length; i++) {

			double[] predict_result = l.predict(test_docs[i], phi);// predict

			int predict_label = Common.maxIndex(predict_result);

			if (predict_label == test_labels_int[i])
				correct_count++;
		}
		System.out.println("zlabel-labeled-LDA: " + (double) correct_count
				/ test_docs.length);

		/*
		 * Formula-LDA begins
		 */

		FormulaLDA fl = new FormulaLDA(med_doc, V, zset, z_cannot_set,
				constrain_map);

		fl.markovChain(K, alpha, beta, iterations);

		phi = fl.estimatePhi();

		correct_count = 0;

		for (int i = 0; i < test_docs.length; i++) {

			double[] predict_result = l.predict(test_docs[i], phi);// predict

			int predict_label = Common.maxIndex(predict_result);

			if (predict_label == test_labels_int[i])
				correct_count++;
		}
		System.out.println("Formula-LDA: " + (double) correct_count
				/ test_docs.length);

		/*
		 * Naive bayes begins
		 */

		final String f = "src//file//med_doc.arff";

		Instances ins = null;
		File file = new File(f);

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		ins = loader.getDataSet();
		ins.setClassIndex(ins.numAttributes() - 1);


		NaiveBayes n = new NaiveBayes();

		n.buildClassifier(ins);

		correct_count = 0;

		for (int i = 0; i < test_docs.length; i++) {
			int attributeNum = vocab.size();
			Instance instance = new Instance(attributeNum);
			int[] tf = new int[attributeNum];

			for (int j = 0; j < test_docs[i].length; j++) {
				tf[test_docs[i][j]]++;
			}

			for (int j = 0; j < attributeNum; j++) {
				instance.setValue(ins.attribute(j), tf[j]);
			}
			instance.setDataset(ins);

			if (test_labels_int[i] == (int) n.classifyInstance(instance))
				correct_count++;

		}

		System.out.println("Naive-Bayes: " + (double) correct_count
				/ test_docs.length);
		
		

	}

}
