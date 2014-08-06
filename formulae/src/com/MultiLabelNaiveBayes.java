package com;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class MultiLabelNaiveBayes {

	public static void main(String args[]) throws Exception {

		/*
		 * 训练K个二分类器
		 */

		for (int k = 0; k < 28; k++) {

			File file = new File("src//file//svm//topic_enhanced" + k + ".arff");

			ArffLoader loader = new ArffLoader();

			loader.setFile(file);

			Instances ins = loader.getDataSet();
			ins.setClassIndex(ins.numAttributes() - 1);

			NaiveBayes n = new NaiveBayes();

			System.out.println("Start training naive bayes..... " + k);

			n.buildClassifier(ins);

			System.out.println("build complete naive bayes " + k);

			Common.SaveModel(n, "naive_bayes" + k + ".model");

		}

		// test on training set
		File file = new File("src//file//svm//topic_enhanced0.arff");

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		Instances ins = loader.getDataSet();

		ins.setClassIndex(ins.numAttributes() - 1);

		int total_length = ins.numInstances();

		List<NaiveBayes> bayes = new ArrayList<NaiveBayes>();

		for (int k = 0; k < 28; k++) {
			NaiveBayes n = (NaiveBayes) Common
					.LoadModel("src//file//svm//model//naive_bayes" + k
							+ ".model");
			bayes.add(n);
		}


		List<Set<Integer>> predict_docs_labels = new ArrayList<Set<Integer>>();

		for (int i = 0; i < total_length; i++) {

			Instance testInst = ins.instance(i);

			System.out.println(i + ":");

			Set<Integer> predict_doc_labels = new HashSet<Integer>();

			for (int k = 0; k < 28; k++) {

				NaiveBayes n = bayes.get(k);
				int predict_result = (int) n.classifyInstance(testInst);
				if (predict_result == 1)
					predict_doc_labels.add(k);
				System.out.print(predict_result + " ");
			}
			predict_docs_labels.add(predict_doc_labels);
			System.out.println();

		}

		/*
		 * F-measure
		 */

		MultiLabelClinicalCases mcc = Preprocess.getMultiLabelClinical();

		// Gold truth
		List<List<String>> docs_labels = mcc.multilabels;

		List<String> label_set = mcc.label_set;

		List<Set<Integer>> real_docs_labels = new ArrayList<Set<Integer>>();

		for (List<String> doc_labels : docs_labels) {

			Set<Integer> real_doc_labels = new HashSet<Integer>();

			for (String label : doc_labels) {

				real_doc_labels.add(label_set.indexOf(label));
			}
			real_docs_labels.add(real_doc_labels);

		}

		// Compare
		int precision_count = 0, total_prediction = 0, total_real = 0;
		for (int i = 0; i < total_length; i++) {

			Set<Integer> predict = predict_docs_labels.get(i);
			Set<Integer> real = real_docs_labels.get(i);

			total_prediction += predict.size();

			total_real += real.size();

			for (int e : predict) {
				if (real.contains(e))
					precision_count++;
			}

		}
		double precision = (double) precision_count / total_prediction;
		double recall = (double) precision_count / total_real;
		double f_measure = 2 * precision * recall / (precision + recall);
		System.out.println("precision : " + precision);
		System.out.println("recall : " + recall);
		System.out.println("f_measure : " + f_measure);

//		Evaluator.five_fold(real_docs_labels);
	}

}
