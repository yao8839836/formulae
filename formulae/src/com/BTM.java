package com;

import java.util.ArrayList;
import java.util.List;

/*
 * Biterm Topic Model WWW'2013
 */
public class BTM {

	int[][] documents;

	int V;

	int K;

	int iterations;

	double[][] alpha; // 每个biterm一个 先验，从label获得

	double beta;

	int[] nb_z; // n(b|z), size K*1

	int[][] nw;

	int[] nwsum;

	int[] label;

	List<Biterm> biterms;

	public BTM(int[][] documents, int V, int[] label) {

		this.documents = documents;
		this.V = V;
		this.label = label;
	}

	public void initialState() {

		biterms = new ArrayList<Biterm>();

		for (int i = 0; i < documents.length; i++) {

			if (documents[i].length >= 2) {

				double weight = 1;
				for (int j = 0; j < documents[i].length - 1; j++)
					for (int k = j + 1; k < documents[i].length; k++) {
						Biterm biterm = new Biterm(documents[i][j],
								documents[i][k], weight, i);
						if (!biterms.contains(biterm))
							biterms.add(biterm);
					}

			}
		}

		nw = new int[V][K];
		nwsum = new int[K];
		nb_z = new int[K];
		alpha = new double[biterms.size()][K];

		System.out.println(biterms.size());

		/*
		 * 随机为每个biterm分配主题
		 */
		for (int i = 0, B = biterms.size(); i < B; i++) {

			int topic = (int) (Math.random() * K);
			Biterm biterm = biterms.get(i);
			int w1 = biterm.wi;
			int w2 = biterm.wj;
			double weight = biterm.weight;
			biterm.z = topic;

			nb_z[topic] += weight;
			nw[w1][topic] += weight;
			nw[w2][topic] += weight;
			nwsum[topic] += weight;

		}

		for (int i = 0; i < documents.length; i++) {

			if (documents[i].length >= 2) {

				double weight = 1;
				for (int j = 0; j < documents[i].length - 1; j++)
					for (int k = j + 1; k < documents[i].length; k++) {
						Biterm biterm = new Biterm(documents[i][j],
								documents[i][k], weight, i);

						alpha[biterms.indexOf(biterm)][label[i]] += (double) 50
								/ K; // 相当于labeled-LDA
//						for (int kl = 0; kl < K; k++)
//							alpha[biterms.indexOf(biterm)][kl] = (double) 50/K;

					}

			}
		}
		
		

	}

	public void markovChain(int K, double beta, int iterations) {

		this.K = K;
		this.beta = beta;
		this.iterations = iterations;

		initialState();

		for (int i = 0; i < this.iterations; i++) {
			// System.out.println(i);
			gibbs();
		}
	}

	public void gibbs() {

		for (int b = 0, B = biterms.size(); b < B; b++) {

			sampleFullConditional(b);

		}
	}

	void sampleFullConditional(int b) {

		Biterm biterm = biterms.get(b);

		int B = biterms.size();

		int topic = biterm.z;
		int w1 = biterm.wi;
		int w2 = biterm.wj;
		double weight = biterm.weight;

		nb_z[topic] -= weight;
		nw[w1][topic] -= weight;
		nw[w2][topic] -= weight;
		nwsum[topic] -= weight;

		double[] p = new double[K];

		double alpha_sum = 0.0;
		for (int i = 0; i < K; i++)
			alpha_sum += alpha[b][i];

		for (int k = 0; k < K; k++) {
			double deno = (double) V / (2 * nb_z[k] + V * beta);
			p[k] = (nb_z[k] + alpha[b][k]) / (B + K * alpha_sum)
					* (nw[w1][k] + beta) * deno * (nw[w2][k] + beta) * deno;
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

		nb_z[topic] += weight;
		nw[w1][topic] += weight;
		nw[w2][topic] += weight;
		nwsum[topic] += weight;

		biterm.z = topic;

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

	// predict label (topic) using bayes rule
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

	// public double[] estimateTheta() {
	//
	// double[] theta = new double [K];
	//		
	// int B = biterms.size();
	//		
	// for(int k = 0; k < K; k++){
	// theta [k] = (nb_z[k] + alpha)/(B + K *alpha);
	// }
	//		
	// return theta;
	// }

	/*
	 * p(z|d) = \sum_w{ p(z|w)p(w|d) }
	 */
	// public double [] doc_infer_sum_w (int [] doc, double [][] phi, double []
	// theta){
	//		
	// double [] prob = new double [K];
	//		
	// for (int i = 0; i < doc.length; i++){
	//			
	// // compute p(z|w) \propo p(w|z)p(z)
	// double [] pz_w = new double [K];
	//			
	// for (int k = 0; k < K; k++){
	// pz_w [k] = theta [k] * phi [k][doc[i]];
	// prob [k] += pz_w [k];
	// }
	//			
	// }
	// return prob;
	// }
	/*
	 * p(z|d) = \sum_b{ p(z|b)p(b|d) } 每个b不一样 还是要按这个
	 */

}

class Biterm {

	int wi;
	int wj;
	int z; // topic assignment

	double weight;

	int doc_num;

	public Biterm(int w1, int w2, double w, int doc) {

		if (w1 <= w2) {
			wi = w1;
			wj = w2;
		} else {
			wi = w2;
			wj = w1;
		}

		z = -1;
		weight = w;

		doc_num = doc;

	}

	@Override
	public String toString() {
		return wi + "," + wj;
	}

	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		if (!(o instanceof Biterm))
			return false;
		Biterm bt = (Biterm) o;
		return bt.wi == wi && bt.wj == wj;
	}

	@Override
	public int hashCode() {
		return this.toString().hashCode();
	}
}