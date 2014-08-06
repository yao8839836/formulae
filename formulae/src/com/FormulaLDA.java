package com;

import java.util.Map;
import java.util.Set;

/*
 * FormulaLDA
 * @author : Liang Yao
 * 
 */

public class FormulaLDA {

	int[][] documents;

	int V;

	int K; // 主题数，等于标签集合的大小

	double[][] alpha; // labeledLDA调整alpha

	double beta;

	int[][] z;

	int[][] nw;

	int[][] nd;

	int[] nwsum;

	int[] ndsum;

	int iterations;

	Map<Integer, Set<Integer>> zsets; // zlabelLDA

	Map<Integer, Set<Integer>> z_cannot_sets;

	Map<Integer, MedConstraint> constraint; // Constrained-LDA 约束

	public FormulaLDA(int[][] docs, int V, Map<Integer, Set<Integer>> zsets,
			Map<Integer, Set<Integer>> z_cannot_sets,
			Map<Integer, MedConstraint> constraint) {

		this.documents = docs;
		this.V = V;
		this.zsets = zsets;
		this.z_cannot_sets = z_cannot_sets;
		this.constraint = constraint;

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

		/*
		 * must-link and cannot link restriction
		 */
		int v = documents[d][n];

		double[] q_score = new double[K];

		if (constraint.get(v).must_link.isEmpty()
				&& constraint.get(v).cannot_link.isEmpty()) {
			for (int k = 0; k < K; k++)
				q_score[k] = 1;
		} else {
			q_score = compute_q(v);
		}

		for (int k = 0; k < K; k++) {

			p[k] = q_score[k] * (nd[d][k] + alpha[d][k])
					/ (ndsum[d] + alpha_sum) * (nw[documents[d][n]][k] + beta)
					/ (nwsum[k] + V * beta);

			// zset resriction

//			int vocab_index = documents[d][n];
//
//			Set<Integer> zset =  zsets.get(vocab_index);
//
//			int ok = 0;
//
//			// Effective Java 推荐
//			for (int e : zset) {
//				if (k == e) {
//					ok = 1;
//					// System.out.println(p[k]);
//					break;
//				}
//			}
//			if (ok == 0) {
//				p[k] = p[k] * (1 - 0.8);
//
//			}

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

	double[] compute_q(int v) {

		double[] q_score = new double[K];

		int[] must_weight = new int[K];

		for (int k = 0; k < K; k++) {

			for (int e : constraint.get(v).must_link) {
				must_weight[k] += nw[e][k];
			}
		}
		// Step 3 - Aggregate
		int[] cannot_weight = new int[K];

		for (int k = 0; k < K; k++) {

			for (int e : constraint.get(v).cannot_link) {
				cannot_weight[k] += nw[e][k];
			}
		}

		double lamada = 0.3;

		for (int k = 0; k < K; k++) {
			if (zsets.get(v).contains(k)) {

				q_score[k] += lamada * must_weight[k];
			}
			if (z_cannot_sets.get(v).contains(k)) {

				q_score[k] -= (1 - lamada) * cannot_weight[k];
			}
		}

		// Step 4 - Normalize and relax

		double max = q_score[Common.maxIndex(q_score)];

		double min = q_score[Common.minIndex(q_score)];

		double eta = 0.9;

		for (int k = 0; k < K; k++) {
			if (max != min) {
				q_score[k] = (q_score[k] - min) / (max - min);
				q_score[k] = q_score[k] * eta + (1 - eta);
			} else {
				q_score[k] = q_score[k] * eta + (1 - eta);
			}

		}

		return q_score;
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
