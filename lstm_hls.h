#pragma once
#include "hls_math.h"
#include "hls_stream.h"
template<typename T>
const T sigmoid(const T value){
#pragma HLS INLINE
	return T(1)/(T(1) + expf(-value));
}
template<typename T>
const T tanh(const T value){
#pragma HLS INLINE
	T v2 = expf(2*value);
	return (v2-T(1))/(T(1) + v2);
}

template<int H, int W, typename T>
void matMul_lstm(const T w[W][H<<2], const T v[W], T o[H<<2]){
#pragma HLS ARRAY_PARTITION variable=w block factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=o block factor=4 dim=1
	for(int i=0;i<H;i++)
#pragma HLS PIPELINE
		for(int j=0;j<4;j++)
			o[(i<<2)+j] = T(0);

	for(int j=0;j<W;j++){
		T r = v[j];
		for(int i=0;i<H;i++)
#pragma HLS PIPELINE
			for(int k=0; k<4; k++)
				o[(i<<2) + k]+=w[j][(i<<2)+k] * r;
	}
}
#if 1
template<int H, int W, typename T>
void matMul(const T w[W][H], const T v[W], T o[H]){
	for(int i=0;i<H;i++)
#pragma HLS PIPELINE
		o[i] = T(0);
	for(int j=0;j<W;j++){
		T tmp = v[j];
		for(int i=0;i<H;i++)
#pragma HLS PIPELINE
			o[i]+=w[j][i] * tmp;
	}
}
#else
template<int H, int W, typename T>
void matMul(const T w[W][H], const T v[W], T o[H]){
#pragma HLS ARRAY_PARTITION variable=w complete dim=1
#pragma HLS ARRAY_PARTITION variable=v complete dim=1

	for(int i=0;i<H;i++){
#pragma HLS PIPELINE
		T tmp = 0;
		for(int j=0;j<W;j++)
			tmp+=w[j][i] * v[i];
		o[i] = tmp;
	}
}
#endif

template<int N, typename T>
void add(const T a0[N], const T a1[N], T a2[N]){
	for(int i=0;i<N;i++)
#pragma HLS PIPELINE
		a2[i] = a0[i] + a1[i] ;
}

template<int N, typename T>
void add(const T a0[N], const T a1[N], const T a2[N], T a3[N]){
	for(int i=0;i<N;i++)
#pragma HLS PIPELINE
		a3[i] = a0[i] + a1[i] + a2[i];
}

template<int INPUT_SIZE, int LSTM_SIZE, typename T>
void lstm(const T input[][INPUT_SIZE],
		hls::stream<T> &output,
		const T BIAS[LSTM_SIZE * 4],
		const T WI[INPUT_SIZE][LSTM_SIZE * 4],
		const T WS[LSTM_SIZE][LSTM_SIZE * 4],
		const int REP){
#pragma HLS ARRAY_PARTITION variable=BIAS block factor=2
#pragma HLS ARRAY_PARTITION variable=WI block factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=WI complete dim=2
#pragma HLS ARRAY_PARTITION variable=WS block factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=WS complete dim=2
	T state[LSTM_SIZE] = {0};
	T hidden[LSTM_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
	for(int rep=0; rep<REP;rep++){
		T in[INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
		T out[LSTM_SIZE];
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
		for(int i =0;i<INPUT_SIZE;i++)
#pragma HLS PIPELINE
			in[i]=input[rep][i];

		for(int i = 0; i < LSTM_SIZE; i++) {
#pragma HLS PIPELINE
			T g0 = BIAS[i];
			T g1 = BIAS[1 * LSTM_SIZE + i];
			T g2 = BIAS[2 * LSTM_SIZE + i];
			T g3 = BIAS[3 * LSTM_SIZE + i];
			for(int j=0;j<INPUT_SIZE;j++){
				g0 +=WI[j][i + (LSTM_SIZE * 0)] * in[j];
				g1 +=WI[j][i + (LSTM_SIZE * 1)] * in[j];
				g2 +=WI[j][i + (LSTM_SIZE * 2)] * in[j];
				g3 +=WI[j][i + (LSTM_SIZE * 3)] * in[j];
			}
			for(int j=0;j<LSTM_SIZE;j++){
				g0 +=WS[j][i + (LSTM_SIZE * 0)] * hidden[j];
				g1 +=WS[j][i + (LSTM_SIZE * 1)] * hidden[j];
				g2 +=WS[j][i + (LSTM_SIZE * 2)] * hidden[j];
				g3 +=WS[j][i + (LSTM_SIZE * 3)] * hidden[j];
			}

#pragma HLS ALLOCATION instances=fexp limit=1 operation
			T s = sigmoid(g1) * state[i] + sigmoid(g0) * tanh(g2);
			state[i] = s;
			out[i] = sigmoid(g3) * tanh(s);
			output.write(out[i]);
		}
		for(int i = 0; i < LSTM_SIZE; i++)
#pragma HLS PIPELINE
			hidden[i] = out[i];
	}
}


template<int INPUT_SIZE, int OUTPUT_SIZE, typename T>
void feed_forward(hls::stream<T> &in,
		T output[][OUTPUT_SIZE],
		const T W1[INPUT_SIZE][OUTPUT_SIZE],
		const T BIAS1[OUTPUT_SIZE],
		const int REP){
#pragma HLS INLINE off
	T tmp[OUTPUT_SIZE];
	T input[INPUT_SIZE];
	for(int rep=0; rep<REP; rep++){
#pragma HLS DATAFLOW
		for(int i=0;i<INPUT_SIZE;i++)
#pragma HLS PIPELINE
			input[i] = in.read();
		matMul<OUTPUT_SIZE, INPUT_SIZE>(W1, input, tmp);
		add<OUTPUT_SIZE>(tmp, BIAS1, output[rep]);
	}
}

