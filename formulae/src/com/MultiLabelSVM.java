package com;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class MultiLabelSVM {

	/*
	 * write arff file for weka
	 */

	public static void write_file(int[][] med_doc,
			List<List<Integer>> doc_labels, int K, int V) throws IOException {

		File file;
		OutputStream out;
		BufferedWriter bw;

		for (int k = 0; k < K; k++) {

			file = new File("src//file//svm//topic" + k + ".arff");
			out = new FileOutputStream(file, false);
			bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));

			bw.write("@relation medicine\n");
			bw.newLine();

			for (int v = 0; v < V; v++) {
				// bw.write("@attribute medicine" + v + " real\n");
				bw.write("@attribute medicine" + v + " real\n");
			}
			bw.write("@attribute function ");
			bw.write("{");

			StringBuilder s = new StringBuilder();
			// K个二分类问题
			for (int k1 = 0; k1 < 2; k1++)
				s.append(k1 + ", ");
			String str = s.substring(0, s.length() - 2);
			bw.write(str);
			bw.write("}\n");
			bw.newLine();

			bw.write("@data\n");

			for (int d = 0; d < med_doc.length; d++) {

				int[] tf = new int[V];
				for (int n = 0, Nd = med_doc[d].length; n < Nd; n++) {
					tf[med_doc[d][n]]++;
				}
				s = new StringBuilder();
				for (int e : tf) {
					s.append(e + ",");
				}
				if (doc_labels.get(d).contains(k))
					s.append(1);
				else
					s.append(0);
				bw.write(s + "\n");
			}
			bw.close();
			out.close();
		}

	}

	public static void write_file(int[][] med_doc,
			List<List<Integer>> doc_labels, int K, int V, double[][] theta)
			throws IOException {

		File file;
		OutputStream out;
		BufferedWriter bw;

		for (int k = 0; k < K; k++) {

			file = new File("src//file//svm//topic_enhanced" + k + ".arff");
			out = new FileOutputStream(file, false);
			bw = new BufferedWriter(new OutputStreamWriter(out, "utf-8"));

			bw.write("@relation medicine\n");
			bw.newLine();

			for (int v = 0; v < V; v++) {
				// bw.write("@attribute medicine" + v + " real\n");
				bw.write("@attribute medicine" + v + " real\n");
			}
			for (int k1 = 0; k1 < K; k1++) {

				bw.write("@attribute topic" + k1 + " real\n");
			}

			bw.write("@attribute function ");
			bw.write("{");

			StringBuilder s = new StringBuilder();
			// K个二分类问题
			for (int k1 = 0; k1 < 2; k1++)
				s.append(k1 + ", ");
			String str = s.substring(0, s.length() - 2);
			bw.write(str);
			bw.write("}\n");
			bw.newLine();

			bw.write("@data\n");

			for (int d = 0; d < med_doc.length; d++) {

				int[] tf = new int[V];
				for (int n = 0, Nd = med_doc[d].length; n < Nd; n++) {
					tf[med_doc[d][n]]++;
				}
				s = new StringBuilder();
				for (int e : tf) {
					s.append(e + ",");
				}
				for (double e : theta[d]) {
					s.append(e + ",");
				}
				if (doc_labels.get(d).contains(k))
					s.append(1);
				else
					s.append(0);
				bw.write(s + "\n");
			}
			bw.close();
			out.close();
		}

	}

	public static void main(String args[]) throws Exception {

		/*
		 * 训练K个二分类器
		 */

		for (int k = 0; k < 28; k++) {

			File file = new File("src//file//svm//topic_only" + k + ".arff");

			ArffLoader loader = new ArffLoader();

			loader.setFile(file);

			Instances ins = loader.getDataSet();
			ins.setClassIndex(ins.numAttributes() - 1);

			LibSVM libsvm = new LibSVM();

			libsvm.setCost(100);

			System.out.println("Start training....." + k);

			libsvm.buildClassifier(ins);

			System.out.println("build complete " + k);

			Common.SaveModel(libsvm, "svm" + k + ".model");
		}

		// test on training set
		File file = new File("src//file//svm//topic_only0.arff");

		ArffLoader loader = new ArffLoader();

		loader.setFile(file);

		Instances ins = loader.getDataSet();

		ins.setClassIndex(ins.numAttributes() - 1);

		int total_length = ins.numInstances();

		List<LibSVM> svms = new ArrayList<LibSVM>();

		for (int k = 0; k < 28; k++) {
			LibSVM libsvm = (LibSVM) Common
					.LoadModel("src//file//svm//model//svm" + k + ".model");
			svms.add(libsvm);
		}

		List<Set<Integer>> predict_docs_labels = new ArrayList<Set<Integer>>();

		for (int i = 0; i < total_length; i++) {

			Instance testInst = ins.instance(i);

			System.out.println(i + ":");

			Set<Integer> predict_doc_labels = new HashSet<Integer>();

			for (int k = 0; k < 28; k++) {

				LibSVM libsvm = svms.get(k);
				int predict_result = (int) libsvm.classifyInstance(testInst);
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

		Evaluator.ten_fold(real_docs_labels);

	}

}
