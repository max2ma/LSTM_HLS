#pragma once
#include "hls_math.h"
#include "hls_stream.h"
template<typename T>
const T sigmoid(const T value){
	return T(1)/(T(1) + expf(-value));
}
template<typename T>
const T tanh(const T value){
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
//#pragma HLS ARRAY_PARTITION variable=w complete dim=2
//#pragma HLS ARRAY_PARTITION variable=o complete dim=1
	for(int i=0;i<H;i++)
#pragma HLS UNROLL
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

template<int LSTM_SIZE, typename T>
void gate_operation(T out1[LSTM_SIZE * 4], T out2[LSTM_SIZE * 4],
		const T BIAS[LSTM_SIZE * 4],
		T hidden[LSTM_SIZE],
		hls::stream<T>& output) {
	static T state[LSTM_SIZE] = {0};
#pragma HLS ALLOCATION instances=tanhf limit=1 operation
	for (int i = 0; i < LSTM_SIZE; i++) {
#pragma HLS PIPELINE
		T g0 = out1[i] + out2[i] + BIAS[i];
		T g1 = out1[LSTM_SIZE + i] + out2[LSTM_SIZE + i] + BIAS[LSTM_SIZE + i];
		T g2 = out1[2 * LSTM_SIZE + i] + out2[2 * LSTM_SIZE + i]
				+ BIAS[2 * LSTM_SIZE + i];
		T g3 = out1[3 * LSTM_SIZE + i] + out2[3 * LSTM_SIZE + i]
				+ BIAS[3 * LSTM_SIZE + i];
		T s = sigmoid(g1) * state[i] + sigmoid(g0) * tanh(g2);
		state[i] = s;
		hidden[i] = sigmoid(g3) * tanh(s);
		output.write(hidden[i]);
	}
}

template<int LSTM_SIZE, int INPUT_SIZE, typename T>
void lstm_dataflow(const T WI[INPUT_SIZE][LSTM_SIZE * 4], const T input[INPUT_SIZE],
		const int rep,  const T WS[LSTM_SIZE][LSTM_SIZE * 4],
		const T hidden[LSTM_SIZE],	const T BIAS[LSTM_SIZE * 4],
		T out[LSTM_SIZE],
		hls::stream<T>& output) {
#pragma HLS DATAFLOW
	T out1[LSTM_SIZE * 4];
	T out2[LSTM_SIZE * 4];
	matMul_lstm<LSTM_SIZE, INPUT_SIZE>(WI, input, out1);
	matMul_lstm<LSTM_SIZE, LSTM_SIZE>(WS, hidden, out2);
	gate_operation<LSTM_SIZE>(out1, out2, BIAS, out, output);
}

template<int INPUT_SIZE, int LSTM_SIZE, typename T>
void lstm(const T input[][INPUT_SIZE],
		hls::stream<T> &output,
		const T BIAS[LSTM_SIZE * 4],
		const T WI[INPUT_SIZE][LSTM_SIZE * 4],
		const T WS[LSTM_SIZE][LSTM_SIZE * 4],
		const int REP){
#pragma HLS ARRAY_PARTITION variable=BIAS block factor=2
	T hidden[LSTM_SIZE] = {0};

	T gates[LSTM_SIZE * 4];
#pragma HLS ARRAY_PARTITION variable=gates block factor=2

	T out[LSTM_SIZE];
	for(int rep=0; rep<REP;rep++){
		lstm_dataflow<LSTM_SIZE, INPUT_SIZE>(WI, input[rep], rep, WS,
				hidden, BIAS, out, output);
		for(int i=0;i<LSTM_SIZE;i++)
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

