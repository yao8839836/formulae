package com;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

public class Common {

	public static int maxIndex(double[] array) {
		int index = 0;
		double max = array[0];
		for (int i = 0; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				index = i;
			}
		}
		return index;
	}

	public static int minIndex(double[] array) {

		int index = 0;

		double min = array[0];

		for (int i = 0; i < array.length; i++) {
			if (array[i] < min) {
				min = array[i];
				index = i;
			}
		}
		return index;

	}

	public static double perplexity(double[][] theta, double[][] phi,
			int[][] docs) {
		double perplexity = 0.0;

		int total_length = 0;
		for (int i = 0; i < docs.length; i++) {
			for (int j = 0; j < docs[i].length; j++)
				total_length++;
		}

		for (int i = 0; i < docs.length; i++) {

			for (int j = 0; j < docs[i].length; j++) {
				double prob = 0.0;
				for (int k = 0; k < phi.length; k++) {
					prob += theta[i][k] * phi[k][docs[i][j]];
				}
				if (prob == 0.0)
					prob = 0.1;
				perplexity += Math.log(prob);

			}
		}

		perplexity = Math.exp(-1 * perplexity / total_length);

		return perplexity;
	}

	public static int[] ArrayRandomSort(int size) {

		int[] positions = new int[size];

		for (int index = 0; index < size; index++) {
			// 初始化数组，以下标为元素值
			positions[index] = index;
		}

		Random random = new Random();

		for (int index = size - 1; index >= 0; index--) {
			// 从0到index处之间随机取一个值，跟index处的元素交换
			int p1 = random.nextInt(index + 1), p2 = index;
			// 交换位置
			int temp = positions[p1];
			positions[p1] = positions[p2];
			positions[p2] = temp;
		}
		// 打印数组的值
		for (int index = 0; index < size; index++) {
			System.out.print(positions[index] + " ");
		}
		System.out.println();

		return positions;

	}

	public static double[] normalize(double[] array) {
		double[] result = new double[array.length];

		double sum = 0.0;

		for (int i = 0; i < array.length; i++) {
			sum += array[i];
		}

		for (int i = 0; i < result.length; i++) {
			result[i] = array[i] / sum;
		}

		return result;
	}

	/**
	 * save the trained classifier to Disk
	 * 
	 * @param classifier
	 *            -the classifier to be saved
	 * @param modelname
	 *            -file name
	 */
	public static void SaveModel(Object classifier, String modelname) {
		try {

			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream("src//file//svm//model//" + modelname));
			oos.writeObject(classifier);
			oos.flush();
			oos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * load the model from disk
	 * 
	 * @param file
	 *            -the model filename
	 * @return-the trained classifier
	 */
	public static Object LoadModel(String file) {
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
					file));
			Object classifier = ois.readObject();
			ois.close();
			return classifier;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			return null;
		}
	}

}
