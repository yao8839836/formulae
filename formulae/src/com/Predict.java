package com;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Predict {

	public static void main(String args[]) throws Exception {

		Corpus c = Preprocess.getCorpus();

		int[][] med_doc_inorder = c.distinct_med_doc;

		String[] med_doc_keys_inorder = c.distinct_map_keys;

		int[] labels_inorder = c.distinct_label_array;

		Map<String, Set<Integer>> label_map = c.label_map;

		// 打乱顺序
		int[] order = Common.ArrayRandomSort(med_doc_inorder.length);

		// 文档顺序
		// for (int i = 0; i < order.length; i++)
		// order[i] = i;

		int[][] med_doc = new int[med_doc_inorder.length][];
		String[] med_doc_keys = new String[med_doc_keys_inorder.length];

		int[] labels = new int[labels_inorder.length];

		for (int d = 0, D = med_doc.length; d < D; d++) {

			int Nd = med_doc_inorder[order[d]].length;
			med_doc[d] = new int[Nd];
			for (int n = 0; n < Nd; n++) {
				med_doc[d][n] = med_doc_inorder[order[d]][n];
			}
			labels[d] = labels_inorder[order[d]];// 有了map之后，只对Naive Bayes 有用
			// med_doc_keys[d] = med_doc_keys_inorder[order[d]];

		}

		List<String> label_set = c.label_set;

		int K = label_set.size();

		List<String> vocab = c.vocab;
		int V = vocab.size();

		double[][] alpha = new double[med_doc.length][K];

		for (int i = 0; i < med_doc.length; i++) {

			alpha[i][labels[i]] = 50 / K;

		}
		double beta = 0.1;

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
		System.out.println(med_doc.length);

		for (int i = 0; i < med_doc.length; i++) {
			for (int j = 0; j < med_doc[i].length; j++)
				System.out.print(med_doc[i][j] + " ");
			System.out.println();
		}
		/*
		 * labeledLDA begins
		 */
		labeledLDA l = new labeledLDA(med_doc, V);

		int iterations = 1000;

		l.markovChain(K, alpha, beta, iterations);

		double[][] theta = l.estimateTheta();
		double[][] phi = l.estimatePhi();

		int[] test_doc = { 175, 61, 248, 520, 8, 38, 24, 42, 56, 14 };
		double[] predict_result = l.predict(test_doc, phi);// predict

		for (int i = 0; i < K; i++)
			System.out.println(predict_result[i]);

		int correct_count = 0;

		for (int i = 0; i < med_doc.length; i++) {

			predict_result = l.predict(med_doc[i], phi);
			int predict_label = Common.maxIndex(predict_result);

			// if (label_map.get(med_doc_keys[i]).contains(predict_label))
			if (predict_label == labels[i])
				correct_count++;
		}
		System.out.println((double) correct_count / med_doc.length);

		System.out.println("labeledLDA perplexity: "
				+ Common.perplexity(theta, phi, med_doc));

		/*
		 * 主题下的high probability medicines
		 */

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + c.label_set.get(k));

			for (int i = 0; i < 20; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");
				phi[k][index] = 0;
			}
			System.out.println();
		}

		/*
		 * 编码相畏相反，相辅相成
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

		// for (int i = 0; i < V; i++) {
		// for (int j = 0; j < V; j++) {
		//
		// if (matrix[i][j] == 0.5) {
		// // 通过相畏相反增加某味药i不能属于的主题（功效）,其他药的主题
		// if (!zset.get(j).isEmpty()) {
		// z_cannot_set.put(i, zset.get(j));
		// // System.out.println(i + " " + zset.get(j));
		//
		// }
		// }
		// }
		// }

		/*
		 * 将zset之外设置成cannot set
		 */
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
		 * labeled-BTM begins
		 */
		BTM btm = new BTM(med_doc, V, labels);

		btm.markovChain(K, beta, iterations);

		phi = btm.estimatePhi();

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + c.label_set.get(k));

			for (int i = 0; i < 20; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");
				phi[k][index] = 0;
			}
			System.out.println();
		}

		// double [] btm_theta = btm.estimateTheta();
		correct_count = 0;

		for (int i = 0; i < med_doc.length; i++) {
			// predict_result = btm.doc_infer_sum_w(med_doc[i], phi, btm_theta);

			predict_result = btm.predict(med_doc[i], phi);
			int predict_label = Common.maxIndex(predict_result);

			// if (label_map.get(med_doc_keys[i]).contains(predict_label))
			if (predict_label == labels[i])
				correct_count++;
		}
		System.out.println("labeled-btm: " + (double) correct_count
				/ med_doc.length);

		/*
		 * labeled-zlabelLDA begins
		 */

		ZsetLabeledLDA zll = new ZsetLabeledLDA(med_doc, V, zset);

		zll.markovChain(K, alpha, beta, iterations);

		theta = zll.estimateTheta();

		phi = zll.estimatePhi();

		correct_count = 0;

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + c.label_set.get(k));

			for (int i = 0; i < 20; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");
				phi[k][index] = 0;
			}
			System.out.println();
		}

		for (int i = 0; i < med_doc.length; i++) {
			predict_result = zll.predict(med_doc[i], phi);
			int predict_label = Common.maxIndex(predict_result);

			// if (label_map.get(med_doc_keys[i]).contains(predict_label))
			if (predict_label == labels[i])
				correct_count++;
		}
		System.out.println((double) correct_count / med_doc.length);

		System.out.println("zlabel-labeledLDA perplexity: "
				+ Common.perplexity(theta, phi, med_doc));

		/*
		 * Formula-LDA begins
		 */

		FormulaLDA fl = new FormulaLDA(med_doc, V, zset, z_cannot_set,
				constrain_map);

		fl.markovChain(K, alpha, beta, iterations);

		theta = fl.estimateTheta();

		phi = fl.estimatePhi();

		for (int k = 0; k < K; k++) {

			System.out.print("topic " + k + ": " + c.label_set.get(k));

			for (int i = 0; i < 20; i++) {
				int index = Common.maxIndex(phi[k]);
				System.out.print(vocab.get(index) + " ");
				phi[k][index] = 0;
			}
			System.out.println();
		}

		correct_count = 0;

		for (int i = 0; i < med_doc.length; i++) {
			predict_result = fl.predict(med_doc[i], phi);
			int predict_label = Common.maxIndex(predict_result);

			// if (label_map.get(med_doc_keys[i]).contains(predict_label))
			// correct_count++;

			if (predict_label == labels[i])
				correct_count++;
		}
		System.out.println((double) correct_count / med_doc.length);

		System.out.println("FomulaLDA perplexity: "
				+ Common.perplexity(theta, phi, med_doc));

		/*
		 * 十折交叉验证
		 */

		order = Common.ArrayRandomSort(c.distinct_med_doc.length);// Naive
		// bayes长度不一致
		new NaiveBayesPredict(c, "src//file//med_doc_same_order.arff", order);

		Evaluator.ten_fold(med_doc, K, V, med_doc_keys, label_map, labels);

		Evaluator
				.ten_fold(med_doc, K, V, zset, med_doc_keys, label_map, labels);

		Evaluator.ten_fold(med_doc, K, V, zset, z_cannot_set, constrain_map,
				med_doc_keys, label_map, labels);

		Evaluator.btm_ten_fold(med_doc, K, V, labels);
		/*
		 * 留一验证
		 */

		// Evaluator.leave_one(med_doc, labels, K, V);
		//        
		// Evaluator.leave_one(med_doc, labels, K, V, zset);
		//        
		// Evaluator.leave_one(med_doc, labels, K, V, zset, z_cannot_set,
		// constrain_map);

	}

}
