package com;

import java.util.Map;
import java.util.Set;

public class ZsetLabeledLDA {

	int[][] documents;

	int V;

	int K; // 主题数，等于标签集合的大小

	double[][] alpha;

	double beta;

	int[][] z;

	int[][] nw;

	int[][] nd;

	int[] nwsum;

	int[] ndsum;

	int iterations;

	Map<Integer, Set<Integer>> zsets;

	public ZsetLabeledLDA(int[][] docs, int V, Map<Integer, Set<Integer>> zsets) {
		this.documents = docs;
		this.V = V;
		this.zsets = zsets;

	}

	public void initialState() {

		int D = documents.length;
		nw = new int[V][K];
		nd = new int[D][K];
		nwsum = new int[K];
		ndsum = new int[D];

		z = new int[D][];
		for (int d = 0; d < D; d++) {

			int Nd = documents[d].length;

			z[d] = new int[Nd];

			for (int n = 0; n < Nd; n++) {
				int topic = (int) (Math.random() * K);

				z[d][n] = topic;

				nw[documents[d][n]][topic]++;

				nd[d][topic]++;

				nwsum[topic]++;
			}
			ndsum[d] = Nd;
		}

	}

	public void markovChain(int K, double[][] alpha, double beta, int iterations) {

		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.iterations = iterations;

		initialState();

		for (int i = 0; i < this.iterations; i++) {
			// System.out.println("iterations: "+i);
			gibbs();
		}
	}

	public void gibbs() {

		for (int d = 0; d < z.length; d++) {

			for (int n = 0; n < z[d].length; n++) {

				int topic = sampleFullConditional(d, n);
				z[d][n] = topic;

			}
		}
	}

	int sampleFullConditional(int d, int n) {

		int topic = z[d][n];
		nw[documents[d][n]][topic]--;
		nd[d][topic]--;
		nwsum[topic]--;
		ndsum[d]--;

		double[] p = new double[K];

		double alpha_sum = 0.0;

		for (int i = 0; i < K; i++)
			alpha_sum += alpha[d][i];

		for (int k = 0; k < K; k++) {

			p[k] = (nd[d][k] + alpha[d][k]) / (ndsum[d] + alpha_sum)
					* (nw[documents[d][n]][k] + beta) / (nwsum[k] + V * beta);

			// zset resriction

			int vocab_index = documents[d][n];

			Set<Integer> zset = zsets.get(vocab_index);

			int ok = 0;

			// Effective Java 推荐
			for (int e : zset) {
				if (k == e) {
					ok = 1;
					// System.out.println(p[k]);
					break;
				}
			}
			if (ok == 0) {
				p[k] = p[k] * (1 - 0.95);

			}

		}
		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1];
		for (int t = 0; t < K; t++) {
			if (u < p[t]) {
				topic = t;
				break;
			}
		}
		nw[documents[d][n]][topic]++;
		nd[d][topic]++;
		nwsum[topic]++;
		ndsum[d]++;
		return topic;

	}

	public double[][] estimateTheta() {
		double[][] theta = new double[documents.length][K];
		for (int d = 0; d < documents.length; d++) {
			double alpha_sum = 0.0;
			for (int k = 0; k < K; k++)
				alpha_sum += alpha[d][k];

			for (int k = 0; k < K; k++) {
				theta[d][k] = (nd[d][k] + alpha[d][k]) / (ndsum[d] + alpha_sum);
			}
		}
		return theta;
	}

	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
		return phi;
	}

	// predict label (topic) using naive bayes
	public double[] predict(int[] doc, double[][] phi) {
		double[] prob = new double[K];

		for (int k = 0; k < K; k++) {
			double product = 1.0;
			for (int i = 0; i < doc.length; i++) {
				product *= phi[k][doc[i]];
			}
			prob[k] = product;
		}
		return prob;

	}

}
